"""Fixes stuff in desc.plotting."""

import numbers
import tkinter
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib import cycler, rcParams
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from packaging.version import Version
from pylatexenc.latex2text import LatexNodes2Text

from desc.backend import OMEGA_IS_0
from desc.compute import data_index, get_transforms
from desc.compute.utils import _parse_parameterization
from desc.equilibrium.coords import map_coordinates
from desc.grid import Grid, LinearGrid
from desc.utils import errorif, only1, parse_argname_change, setdefault

__all__ = ["plot_2d", "plot_3d", "plot_comparison", "plot_section", "plot_surfaces"]


colorblind_colors = [
    (0.0000, 0.4500, 0.7000),  # blue
    (0.8359, 0.3682, 0.0000),  # vermilion
    (0.0000, 0.6000, 0.5000),  # bluish green
    (0.9500, 0.9000, 0.2500),  # yellow
    (0.3500, 0.7000, 0.9000),  # sky blue
    (0.8000, 0.6000, 0.7000),  # reddish purple
    (0.9000, 0.6000, 0.0000),  # orange
]
sequential_colors = [
    "#c80016",  # red
    "#dc5b0e",  # burnt orange
    "#f0b528",  # light orange
    "#dce953",  # yellow
    "#7acf7c",  # green
    "#1fb7c9",  # teal
    "#2192e3",  # medium blue
    "#4f66d4",  # blue-violet
    "#7436a5",  # purple
]
dashes = [
    (1.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # solid
    (3.7, 1.6, 0.0, 0.0, 0.0, 0.0),  # dashed
    (1.0, 1.6, 0.0, 0.0, 0.0, 0.0),  # dotted
    (6.4, 1.6, 1.0, 1.6, 0.0, 0.0),  # dot dash
    (3.0, 1.6, 1.0, 1.6, 1.0, 1.6),  # dot dot dash
    (6.0, 4.0, 0.0, 0.0, 0.0, 0.0),  # long dash
    (1.0, 1.6, 3.0, 1.6, 3.0, 1.6),  # dash dash dot
]
matplotlib.rcdefaults()
rcParams["font.family"] = "DejaVu Serif"
rcParams["mathtext.fontset"] = "cm"
rcParams["font.size"] = 10
rcParams["figure.facecolor"] = (1, 1, 1, 1)
rcParams["figure.figsize"] = (6, 4)

try:
    dpi = tkinter.Tk().winfo_fpixels("1i")
except tkinter._tkinter.TclError:
    dpi = 72
rcParams["figure.dpi"] = dpi
rcParams["figure.autolayout"] = True
rcParams["axes.spines.top"] = False
rcParams["axes.spines.right"] = False
rcParams["axes.labelsize"] = "small"
rcParams["axes.titlesize"] = "medium"
rcParams["lines.linewidth"] = 1
rcParams["lines.solid_capstyle"] = "round"
rcParams["lines.dash_capstyle"] = "round"
rcParams["lines.dash_joinstyle"] = "round"
rcParams["xtick.labelsize"] = "x-small"
rcParams["ytick.labelsize"] = "x-small"
color_cycle = cycler(color=colorblind_colors)
dash_cycle = cycler(dashes=dashes)
rcParams["axes.prop_cycle"] = color_cycle

_AXIS_LABELS_RTZ = [r"$\rho$", r"$\theta$", r"$\zeta$"]
_AXIS_LABELS_RPZ = [r"$R ~(\mathrm{m})$", r"$\phi$", r"$Z ~(\mathrm{m})$"]
_AXIS_LABELS_XYZ = [r"$X ~(\mathrm{m})$", r"$Y ~(\mathrm{m})$", r"$Z ~(\mathrm{m})$"]


def _set_tight_layout(fig):
    return


def _get_cmap(name, n=None):
    # TODO: update this when matplotlib min version >= 3.6.0
    # compat layer to deal with API changes in mpl 3.6.0
    if Version(matplotlib.__version__) >= Version("3.6.0"):
        c = matplotlib.colormaps[name]
        if n is not None:
            c = c.resampled(n)
        return c
    else:
        return matplotlib.cm.get_cmap(name, n)


def _format_ax(ax, is3d=False, rows=1, cols=1, figsize=None, equal=False):
    """Check type of ax argument. If ax is not a matplotlib AxesSubplot, initialize one.

    Parameters
    ----------
    ax : None or matplotlib AxesSubplot instance
        Axis to plot on.
    is3d: bool
        Whether the plot is three-dimensional.
    rows : int, optional
        Number of rows of subplots to create.
    cols : int, optional
        Number of columns of subplots to create.
    figsize : tuple of 2 floats
        Figure size (width, height) in inches. Default is (6, 6).
    equal : bool
        Whether axes should have equal scales for x and y.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure being plotted to.
    ax : matplotlib.axes.Axes or ndarray of Axes
        Axes being plotted to.

    """
    if figsize is None:
        figsize = (6, 6)
    if ax is None:
        if is3d:
            fig = plt.figure(figsize=figsize, dpi=dpi, layout="constrained")
            ax = np.array(
                [
                    fig.add_subplot(rows, cols, int(r * cols + c + 1), projection="3d")
                    for r in range(rows)
                    for c in range(cols)
                ]
            ).reshape((rows, cols))
            if ax.size == 1:
                ax = ax.flatten()[0]
            return fig, ax
        else:
            fig, ax = plt.subplots(
                rows,
                cols,
                figsize=figsize,
                squeeze=False,
                sharex=True,
                sharey=True,
                subplot_kw=dict(aspect="equal") if equal else None,
            )
            if ax.size == 1:
                ax = ax.flatten()[0]
            return fig, ax

    elif isinstance(ax, matplotlib.axes.Axes):
        return plt.gcf(), ax
    else:
        ax = np.atleast_1d(ax)
        errorif(
            not isinstance(ax.flatten()[0], matplotlib.axes.Axes),
            TypeError,
            "ax argument must be None or an axis instance or array of axes",
        )
        return plt.gcf(), ax


def _get_grid(**kwargs):
    """Get grid for plotting.

    Parameters
    ----------
    kwargs
         Any arguments taken by LinearGrid.

    Returns
    -------
    grid : LinearGrid
         Grid of coordinates to evaluate at.

    """
    grid_args = {
        "NFP": 1,
        "sym": False,
        "axis": True,
        "endpoint": True,
    }
    grid_args.update(kwargs)
    if ("L" not in grid_args) and ("rho" not in grid_args):
        grid_args["rho"] = np.array([1.0])
    if ("M" not in grid_args) and ("theta" not in grid_args):
        grid_args["theta"] = np.array([0.0])
    if ("N" not in grid_args) and ("zeta" not in grid_args):
        grid_args["zeta"] = np.array([0.0])

    grid = LinearGrid(**grid_args)

    return grid


def _get_plot_axes(grid):
    """Find which axes are being plotted.

    Parameters
    ----------
    grid : Grid
        Grid of coordinates to evaluate at.

    Returns
    -------
    axes : tuple of int
        Which axes of the grid are being plotted.

    """
    plot_axes = [0, 1, 2]
    if grid.num_rho == 1:
        plot_axes.remove(0)
    if grid.num_theta == 1:
        plot_axes.remove(1)
    if grid.num_zeta == 1:
        plot_axes.remove(2)

    return tuple(plot_axes)


def _compute(
    eq,
    name,
    grid,
    component=None,
    reshape=True,
    compute_kwargs=None,
):
    """Compute quantity specified by name on grid for Equilibrium eq.

    Parameters
    ----------
    eq : Equilibrium
        Object from which to plot.
    name : str
        Name of variable to plot.
    grid : Grid
        Grid of coordinates to evaluate at.
    component : str, optional
        For vector variables, which element to plot. Default is the norm of the vector.
    compute_kwargs : dict, optional
        Additional keyword arguments to pass to ``eq.compute``

    Returns
    -------
    data : float array of shape (M, L, N)
        Computed quantity.

    """
    parameterization = _parse_parameterization(eq)
    errorif(
        name not in data_index[parameterization],
        msg=f"Unrecognized value '{name}' for parameterization {parameterization}.",
    )
    assert component in [
        None,
        "R",
        "phi",
        "Z",
    ], f"component must be one of [None, 'R', 'phi', 'Z'], got {component}"

    components = {"R": 0, "phi": 1, "Z": 2}

    label = data_index[parameterization][name]["label"]

    compute_kwargs = setdefault(compute_kwargs, {})

    data = eq.compute(name, grid=grid, **compute_kwargs)[name]

    if data_index[parameterization][name]["dim"] > 1:
        if component is None:
            data = np.linalg.norm(data, axis=-1)
            label = "|" + label + "|"
        else:
            data = data[:, components[component]]
            label = "(" + label + ")_"
            if component in ["R", "Z"]:
                label += component
            else:
                label += r"\phi"

    units = data_index[parameterization][name]["units"]
    if units == "~":
        label = rf"${label}$"
    else:
        label = r"$" + label + "~(" + units + ")$"

    if reshape:
        data = data.reshape((grid.num_theta, grid.num_rho, grid.num_zeta), order="F")

    return data, label


def plot_2d(  # noqa : C901
    eq,
    name,
    grid=None,
    log=False,
    normalize=None,
    ax=None,
    return_data=False,
    compute_kwargs=None,
    **kwargs,
):
    """Plot 2D cross-sections.

    Parameters
    ----------
    eq : Equilibrium, Surface
        Object from which to plot.
    name : str
        Name of variable to plot.
    grid : Grid, optional
        Grid of coordinates to plot at.
    log : bool, optional
        Whether to use a log scale.
    normalize : str, optional
        Name of the variable to normalize ``name`` by. Default is None.
    ax : matplotlib AxesSubplot, optional
        Axis to plot on.
    return_data : bool
        If True, return the data plotted as well as fig,ax
    compute_kwargs : dict, optional
        Additional keyword arguments to pass to ``eq.compute``.
    **kwargs : dict, optional
        Specify properties of the figure, axis, and plot appearance e.g.::

            plot_X(figsize=(4,6),cmap="plasma")

        Valid keyword arguments are:

        * ``figsize``: tuple of length 2, the size of the figure (to be passed to
          matplotlib)
        * ``component``: str, one of [None, 'R', 'phi', 'Z'], For vector variables,
          which element to plot. Default is the norm of the vector.
        * ``title_fontsize``: integer, font size of the title
        * ``xlabel_fontsize``: float, fontsize of the xlabel
        * ``ylabel_fontsize``: float, fontsize of the ylabel
        * ``cmap``: str, matplotlib colormap scheme to use, passed to ax.contourf
        * ``levels``: int or array-like, passed to contourf.
          If ``name="|F|_normalized"`` and ``log=True``, default is
          ``np.logspace(-6, 0, 7)``. Otherwise the default (``None``) uses the min/max
          values of the data.
        * ``field``: MagneticField, a magnetic field with which to calculate Bn on
          the surface, must be provided if Bn is entered as the variable to plot.
        * ``field_grid``: MagneticField, a Grid to pass to the field as a source grid
          from which to calculate Bn, by default None.
        * ``filled`` : bool, whether to fill contours or not i.e. whether to use
          `contourf` or `contour`

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure being plotted to.
    ax : matplotlib.axes.Axes or ndarray of Axes
        Axes being plotted to.
    plot_data : dict
        Dictionary of the data plotted, only returned if ``return_data=True``

    Examples
    --------
    .. image:: ../../_static/images/plotting/plot_2d.png

    .. code-block:: python

        from desc.plotting import plot_2d
        plot_2d(eq, 'sqrt(g)')

    """
    label = kwargs.pop("label", None)
    normalize = parse_argname_change(normalize, kwargs, "norm_name", "normalize")
    if "norm_F" in kwargs:
        norm_F = kwargs.pop("norm_F")
        warnings.warn(
            FutureWarning(
                "Argument norm_F has been deprecated. If you are trying to "
                + "normalize |F| by magnetic pressure gradient, use  "
                + "`name=|F|_normalized` instead. If you want to normalize by "
                + "another quantity, use the `normalize` keyword argument."
            )
        )
        if normalize is None and norm_F:
            # replicate old behavior before #1683
            normalize = "<|grad(|B|^2)|/2mu0>_vol"
        elif normalize is not None and norm_F:
            raise ValueError("Cannot use both norm_F and normalize keyword arguments.")
    errorif(
        not (isinstance(normalize, str) or normalize is None),
        ValueError,
        "normalize must be a string",
    )
    parameterization = _parse_parameterization(eq)
    if grid is None:
        grid_kwargs = {"M": 33, "N": 33, "NFP": eq.NFP, "axis": False}
        grid = _get_grid(**grid_kwargs)
    plot_axes = _get_plot_axes(grid)
    errorif(len(plot_axes) != 2, msg="Grid must be 2D")
    component = kwargs.pop("component", None)
    if name == "alpha, zeta to theta - alpha":

        iota_grid = LinearGrid(rho=1.0)
        iota = iota_grid.compress(eq.compute("iota", grid=iota_grid)["iota"])
        data = eq._map_poloidal_coordinates(
            iota,
            grid.nodes[grid.unique_poloidal_idx, 1],
            grid.nodes[grid.unique_zeta_idx, 2],
            eq.params_dict["L_lmn"],
            get_transforms("lambda", eq, grid)["L"],
            inbasis="alpha",
            outbasis="delta",
            tol=1e-10,
        ).swapaxes(0, 1)

    elif name != "B*n":
        data, label_default = _compute(
            eq,
            name,
            grid,
            component=component,
            compute_kwargs=compute_kwargs,
        )
        if label is None:
            label = label_default
    else:
        pass

    fig, ax = _format_ax(ax, figsize=kwargs.pop("figsize", None))
    # divider = make_axes_locatable(ax)

    if normalize:
        norm_data, _ = _compute(
            eq,
            normalize,
            grid,
            reshape=False,
            compute_kwargs=compute_kwargs,
        )
        data = data / np.nanmean(np.abs(norm_data))  # normalize

    # reshape data to 2D
    if 0 in plot_axes:
        if 1 in plot_axes:  # rho & theta
            data = data[:, :, 0]
        else:  # rho & zeta
            data = data[0, :, :]
    else:  # theta & zeta
        data = data[:, 0, :]

    contourf_kwargs = {}
    if log:
        data = np.abs(data)  # ensure data is positive for log plot
        contourf_kwargs["norm"] = matplotlib.colors.LogNorm()
        if name == "|F|_normalized" or (
            name == "|F|" and normalize == "<|grad(|B|^2)|/2mu0>_vol"
        ):
            contourf_kwargs["levels"] = kwargs.pop("levels", np.logspace(-6, 0, 7))
        else:
            logmin = max(np.floor(np.nanmin(np.log10(data))).astype(int), -16)
            logmax = np.ceil(np.nanmax(np.log10(data))).astype(int)
            contourf_kwargs["levels"] = kwargs.pop(
                "levels", np.logspace(logmin, logmax, logmax - logmin + 1)
            )
    else:
        contourf_kwargs["norm"] = matplotlib.colors.Normalize()
        contourf_kwargs["levels"] = kwargs.pop("levels", 25)
    contourf_kwargs["cmap"] = kwargs.pop("cmap", "jet")
    title_fontsize = kwargs.pop("title_fontsize", None)
    xlabel_fontsize = kwargs.pop("xlabel_fontsize", None)
    ylabel_fontsize = kwargs.pop("ylabel_fontsize", None)
    filled = kwargs.pop("filled", True)
    cbar_format = kwargs.pop("cbar_format", None)
    cbar_ax_tick_label_size = kwargs.pop("cbar_ax_tick_label_size", None)
    ax_tick_params_label_size = kwargs.pop("ax_tick_params_label_size", None)
    assert len(kwargs) == 0, f"plot_2d got unexpected keyword argument: {kwargs.keys()}"

    # cax_kwargs = {"size": "5%", "pad": 0.05}

    xx = (
        grid.nodes[:, plot_axes[1]]
        .reshape((grid.num_theta, grid.num_rho, grid.num_zeta), order="F")
        .squeeze()
    )
    yy = (
        grid.nodes[:, plot_axes[0]]
        .reshape((grid.num_theta, grid.num_rho, grid.num_zeta), order="F")
        .squeeze()
    )
    if not filled:
        im = ax.contour(xx, yy, data, **contourf_kwargs)
    else:
        im = ax.contourf(xx, yy, data, **contourf_kwargs)
    # cax = divider.append_axes("right", **cax_kwargs)
    if cbar_format is None:
        cbar = fig.colorbar(im)
    else:
        cbar = fig.colorbar(im, format=cbar_format)
    if cbar_ax_tick_label_size is not None:
        cbar.ax.tick_params(labelsize=cbar_ax_tick_label_size)
    cbar.update_ticks()
    xlabel = _AXIS_LABELS_RTZ[plot_axes[1]]
    ylabel = _AXIS_LABELS_RTZ[plot_axes[0]]
    if name == "alpha, zeta to theta - alpha":
        ylabel = r"$\alpha$"
        xlabel = r"$\zeta$"
    ax.set_xlabel(xlabel, fontsize=xlabel_fontsize)
    ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
    ax.set_title(label, fontsize=title_fontsize)
    if ax_tick_params_label_size is not None:
        ax.tick_params(labelsize=ax_tick_params_label_size)
    if normalize:
        ax.set_title(
            "%s / %s"
            % (
                "$" + data_index[parameterization][name]["label"] + "$",
                "$" + data_index[parameterization][normalize]["label"] + "$",
            )
        )
    _set_tight_layout(fig)
    plot_data = {
        xlabel.strip("$").strip("\\"): xx,
        ylabel.strip("$").strip("\\"): yy,
        name: data,
    }

    if normalize:
        plot_data["normalization"] = np.nanmean(np.abs(norm_data))
    else:
        plot_data["normalization"] = 1
    if return_data:
        return fig, ax, plot_data

    return fig, ax


def _trimesh_idx(n1, n2, periodic1=True, periodic2=True):
    # suppose grid is something like this (n1=3, n2=4):
    # 0  1  2  3
    # 4  5  6  7
    # 8  9 10 11

    # first set of triangles are (0,1,4), (1,2,5), (2,3,6), (3,0,7), ... (8,9,0) etc
    # second set are (1,5,4), (2,6,5), (3,7,6), (0,4,7) etc.
    # for the first set, i1 is the linear index, j1 = i1+1, k1=i1+n2
    # for second set, i2 from the second set is j1 from the first, and j2 = k1,
    # k2 = i1 + 1 + n2 with some other tricks to handle wrapping or out of bounds
    n = n1 * n2
    c, r = np.meshgrid(np.arange(n1), np.arange(n2), indexing="ij")

    def clip_or_mod(x, p, flag):
        if flag:
            return x % p
        else:
            return np.clip(x, 0, p - 1)

    i1 = c * n2 + r
    j1 = c * n2 + clip_or_mod((r + 1), n2, periodic2)
    k1 = clip_or_mod((c + 1), n1, periodic1) * n2 + r

    i2 = c * n2 + clip_or_mod((r + 1), n2, periodic2)
    j2 = clip_or_mod((c + 1), n1, periodic1) * n2 + r
    k2 = clip_or_mod((c + 1), n1, periodic1) * n2 + clip_or_mod((r + 1), n2, periodic2)

    i = np.concatenate([i1.flatten(), i2.flatten()])
    j = np.concatenate([j1.flatten(), j2.flatten()])
    k = np.concatenate([k1.flatten(), k2.flatten()])

    # remove degenerate triangles, ie with the same vertex twice
    degens = (i == j) | (j == k) | (i == k)
    ijk = np.array([i, j, k])[:, ~degens]
    # remove out of bounds indices
    ijk = ijk[:, np.all(ijk < n, axis=0)]
    # remove duplicates
    ijk = np.unique(np.sort(ijk, axis=0), axis=1)

    # expected number of triangles
    # start with 2 per square
    exnum = (n1 - 1) * (n2 - 1) * 2
    # if periodic, add extra "ghost" cells to connect ends
    if periodic1:
        exnum += (n2 - 1) * 2
    if periodic2:
        exnum += (n1 - 1) * 2
    # if doubly periodic, there's also 2 at the corner
    if periodic1 and periodic2:
        exnum += 2

    assert ijk.shape[1] == exnum
    return ijk


def plot_3d(  # noqa : C901
    eq,
    name,
    grid=None,
    log=False,
    normalize=None,
    fig=None,
    return_data=False,
    compute_kwargs=None,
    **kwargs,
):
    """Plot 3D surfaces.

    Parameters
    ----------
    eq : Equilibrium, Surface
        Object from which to plot.
    name : str
        Name of variable to plot.
    grid : Grid, optional
        Grid of coordinates to plot at.
    log : bool, optional
        Whether to use a log scale.
    normalize : str, optional
        Name of the variable to normalize ``name`` by. Default is None.
    fig : plotly.graph_objs._figure.Figure, optional
        Figure to plot on.
    return_data : bool
        If True, return the data plotted as well as fig,ax
    compute_kwargs : dict, optional
        Additional keyword arguments to pass to ``eq.compute``.
    **kwargs : dict, optional
        Specify properties of the figure, axis, and plot appearance e.g.::

            plot_X(figsize=(4,6), cmap="RdBu")

        Valid keyword arguments are:

        * ``figsize``: tuple of length 2, the size of the figure in inches
        * ``component``: str, one of [None, 'R', 'phi', 'Z'], For vector variables,
          which element to plot. Default is the norm of the vector.
        * ``title``: title to add to the figure.
        * ``cmap``: string denoting colormap to use.
        * ``levels``: array of data values where ticks on colorbar should be placed.
        * ``alpha``: float in [0,1.0], the transparency of the plotted surface
        * ``showscale``: Bool, whether or not to show the colorbar. True by default.
        * ``showgrid``: Bool, whether or not to show the coordinate grid lines.
          True by default.
        * ``showticklabels``: Bool, whether or not to show the coordinate tick labels.
          True by default.
        * ``showaxislabels``: Bool, whether or not to show the coordinate axis labels.
          True by default.
        * ``zeroline``: Bool, whether or not to show the zero coordinate axis lines.
          True by default.
        * ``field``: MagneticField, a magnetic field with which to calculate Bn on
          the surface, must be provided if Bn is entered as the variable to plot.
        * ``field_grid``: MagneticField, a Grid to pass to the field as a source grid
          from which to calculate Bn, by default None.


    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        Figure being plotted to
    plot_data : dict
        Dictionary of the data plotted, only returned if ``return_data=True``

    Examples
    --------
    .. raw:: html

        <iframe src="../../_static/images/plotting/plot_3d.html"
        width="100%" height="980" frameborder="0"></iframe>

    .. code-block:: python

        from desc.plotting import plot_3d
        from desc.grid import LinearGrid
        grid = LinearGrid(
                rho=0.5,
                theta=np.linspace(0, 2 * np.pi, 100),
                zeta=np.linspace(0, 2 * np.pi, 100),
                axis=True,
            )
        fig = plot_3d(eq, "|F|", log=True, grid=grid)

    """
    errorif(
        not (isinstance(normalize, str) or normalize is None),
        ValueError,
        "normalize must be a string",
    )

    if grid is None:
        grid_kwargs = {"M": 50, "N": int(50 * eq.NFP), "NFP": 1, "endpoint": True}
        grid = _get_grid(**grid_kwargs)
    assert isinstance(grid, LinearGrid), "grid must be LinearGrid for 3d plotting"
    assert only1(
        grid.num_rho == 1, grid.num_theta == 1, grid.num_zeta == 1
    ), "Grid must be 2D"
    figsize = kwargs.pop("figsize", (10, 10))
    alpha = kwargs.pop("alpha", 1.0)
    cmap = kwargs.pop("cmap", "RdBu_r")
    title = kwargs.pop("title", "")
    levels = kwargs.pop("levels", None)
    component = kwargs.pop("component", None)
    showgrid = kwargs.pop("showgrid", True)
    zeroline = kwargs.pop("zeroline", True)
    showscale = kwargs.pop("showscale", True)
    showticklabels = kwargs.pop("showticklabels", True)
    showaxislabels = kwargs.pop("showaxislabels", True)
    update_traces = kwargs.pop("update_traces", None)

    if name != "B*n":
        data, label = _compute(
            eq,
            name,
            grid,
            component=component,
            compute_kwargs=compute_kwargs,
        )
    else:
        pass

    if normalize:
        norm_data, _ = _compute(eq, normalize, grid, reshape=False)
        data = data / np.nanmean(np.abs(norm_data))  # normalize

    errorif(
        len(kwargs) != 0,
        msg=f"plot_3d got unexpected keyword argument: {kwargs.keys()}",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coords = eq.compute(["X", "Y", "Z"], grid=grid)
    X = coords["X"].reshape((grid.num_theta, grid.num_rho, grid.num_zeta), order="F")
    Y = coords["Y"].reshape((grid.num_theta, grid.num_rho, grid.num_zeta), order="F")
    Z = coords["Z"].reshape((grid.num_theta, grid.num_rho, grid.num_zeta), order="F")

    if grid.num_rho == 1:
        n1, n2 = grid.num_theta, grid.num_zeta
        if not grid.nodes[-1][2] == 2 * np.pi:
            p1, p2 = False, False
        else:
            p1, p2 = False, True
    elif grid.num_theta == 1:
        n1, n2 = grid.num_rho, grid.num_zeta
        p1, p2 = False, True
    elif grid.num_zeta == 1:
        n1, n2 = grid.num_theta, grid.num_rho
        p1, p2 = True, False
    ijk = _trimesh_idx(n1, n2, p1, p2)

    if log:
        data = np.log10(np.abs(data))  # ensure data is positive for log plot
        cmin = np.floor(np.nanmin(data)).astype(int)
        cmax = np.ceil(np.nanmax(data)).astype(int)
        levels = setdefault(levels, np.logspace(cmin, cmax, cmax - cmin + 1))
        ticks = np.log10(levels)
        cbar = dict(
            title=LatexNodes2Text().latex_to_text(label),
            ticktext=[f"{l:.0e}" for l in levels],
            tickvals=ticks,
        )
    else:
        cbar = dict(
            title=LatexNodes2Text().latex_to_text(label),
            ticktext=levels,
            tickvals=levels,
        )
        cmin = None
        cmax = None

    if normalize:
        parameterization = _parse_parameterization(eq)
        label = "{} / {}".format(
            "$" + data_index[parameterization][name]["label"] + "$",
            "$" + data_index[parameterization][normalize]["label"] + "$",
        )
        cbar["title"] = LatexNodes2Text().latex_to_text(label)

    meshdata = go.Mesh3d(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        intensity=data.flatten(),
        opacity=alpha,
        cmin=cmin,
        cmax=cmax,
        i=ijk[0],
        j=ijk[1],
        k=ijk[2],
        colorscale=cmap,
        flatshading=True,
        name=LatexNodes2Text().latex_to_text(label),
        colorbar=cbar,
        showscale=showscale,
    )

    if fig is None:
        fig = go.Figure()

    fig.add_trace(meshdata)
    if update_traces is not None:
        fig.update_traces(**update_traces)

    xaxis_title = (
        LatexNodes2Text().latex_to_text(_AXIS_LABELS_XYZ[0]) if showaxislabels else ""
    )
    yaxis_title = (
        LatexNodes2Text().latex_to_text(_AXIS_LABELS_XYZ[1]) if showaxislabels else ""
    )
    zaxis_title = (
        LatexNodes2Text().latex_to_text(_AXIS_LABELS_XYZ[2]) if showaxislabels else ""
    )

    fig.update_layout(
        scene=dict(
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            zaxis_title=zaxis_title,
            aspectmode="data",
            xaxis=dict(
                backgroundcolor="white",
                gridcolor="darkgrey",
                showbackground=False,
                zerolinecolor="darkgrey",
                showgrid=showgrid,
                zeroline=zeroline,
                showticklabels=showticklabels,
            ),
            yaxis=dict(
                backgroundcolor="white",
                gridcolor="darkgrey",
                showbackground=False,
                zerolinecolor="darkgrey",
                showgrid=showgrid,
                zeroline=zeroline,
                showticklabels=showticklabels,
            ),
            zaxis=dict(
                backgroundcolor="white",
                gridcolor="darkgrey",
                showbackground=False,
                zerolinecolor="darkgrey",
                showgrid=showgrid,
                zeroline=zeroline,
                showticklabels=showticklabels,
            ),
        ),
        width=figsize[0] * dpi,
        height=figsize[1] * dpi,
        title=dict(text=title, y=0.9, x=0.5, xanchor="center", yanchor="top"),
        font=dict(family="Times", color="black"),
    )
    plot_data = {"X": X, "Y": Y, "Z": Z, name: data}

    if normalize:
        plot_data["normalization"] = np.nanmean(np.abs(norm_data))
    else:
        plot_data["normalization"] = 1

    if return_data:
        return fig, plot_data

    return fig


def plot_section(
    eq,
    name,
    grid=None,
    log=False,
    normalize=None,
    ax=None,
    return_data=False,
    compute_kwargs=None,
    **kwargs,
):
    """Plot Poincare sections.

    Parameters
    ----------
    eq : Equilibrium
        Object from which to plot.
    name : str
        Name of variable to plot.
    grid : Grid, optional
        Grid of coordinates to plot at.
    log : bool, optional
        Whether to use a log scale.
    normalize : str, optional
        Name of the variable to normalize ``name`` by. Default is None.
    ax : matplotlib AxesSubplot, optional
        Axis to plot on.
    return_data : bool
        If True, return the data plotted as well as fig,ax
    compute_kwargs : dict, optional
        Additional keyword arguments to pass to ``eq.compute``.
    **kwargs : dict, optional
        Specify properties of the figure, axis, and plot appearance e.g.::

            plot_X(figsize=(4,6),label="your_label")

        Valid keyword arguments are:

        * ``figsize``: tuple of length 2, the size of the figure (to be passed to
          matplotlib)
        * ``component``: str, one of [None, 'R', 'phi', 'Z'], For vector variables,
          which element to plot. Default is the norm of the vector.
        * ``title_fontsize``: integer, font size of the title
        * ``xlabel_fontsize``: float, fontsize of the xlabel
        * ``ylabel_fontsize``: float, fontsize of the ylabel
        * ``cmap``: str, matplotlib colormap scheme to use, passed to ax.contourf
        * ``levels``: int or array-like, passed to contourf.
          If ``name="|F|_normalized"`` and ``log=True``, default is
          ``np.logspace(-6, 0, 7)``. Otherwise the default (``None``) uses the min/max
          values of the data.
        * ``phi``: float, int or array-like. Toroidal angles to plot. If an integer,
          plot that number equally spaced in [0,2pi/NFP). Default 1 for axisymmetry and
          6 for non-axisymmetry
        * ``fill`` : bool,  Whether the contours are filled, i.e. whether to use
          `contourf` or `contour`. Default to ``fill=True``

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure being plotted to.
    ax : matplotlib.axes.Axes or ndarray of Axes
        Axes being plotted to.
    plot_data : dict
        Dictionary of the data plotted, only returned if ``return_data=True``

    Examples
    --------
    .. image:: ../../_static/images/plotting/plot_section.png

    .. code-block:: python

        from desc.plotting import plot_section
        fig, ax = plot_section(eq, "J^rho")

    """
    normalize = parse_argname_change(normalize, kwargs, "norm_name", "normalize")
    if "norm_F" in kwargs:
        norm_F = kwargs.pop("norm_F")
        warnings.warn(
            FutureWarning(
                "Argument norm_F has been deprecated. If you are trying to "
                + "normalize |F| by magnetic pressure gradient, use  "
                + "`name=|F|_normalized` instead. If you want to normalize by "
                + "another quantity, use the `normalize` keyword argument."
            )
        )
        if normalize is None and norm_F:
            # replicate old behavior before #1683
            normalize = "<|grad(|B|^2)|/2mu0>_vol"
        elif normalize is not None and norm_F:
            raise ValueError("Cannot use both norm_F and normalize keyword arguments.")
    errorif(
        not (isinstance(normalize, str) or normalize is None),
        ValueError,
        "normalize must be a string",
    )
    phi = kwargs.pop("phi", (1 if eq.N == 0 else 6))
    phi = parse_argname_change(phi, kwargs, "nzeta", "phi")
    phi = parse_argname_change(phi, kwargs, "nphi", "phi")

    if isinstance(phi, numbers.Integral):
        phi = np.linspace(0, 2 * np.pi / eq.NFP, phi, endpoint=False)
    phi = np.atleast_1d(phi)
    nphi = len(phi)
    if grid is None:
        grid_kwargs = {
            "L": max(25, eq.L_grid),
            "NFP": 1,
            "axis": False,
            "theta": np.linspace(0, 2 * np.pi, 91, endpoint=True),
            "zeta": phi,
        }
        grid = _get_grid(**grid_kwargs)
        nr, nt, nz = grid.num_rho, grid.num_theta, grid.num_zeta
        coords = map_coordinates(
            eq,
            grid.nodes,
            ["rho", "theta", "phi"],
            ["rho", "theta", "zeta"],
            period=(np.inf, 2 * np.pi, 2 * np.pi),
            guess=grid.nodes,
        )
        grid = Grid(coords, sort=False)
    else:
        phi = np.unique(grid.nodes[:, 2])
        nphi = phi.size
        nr, nt, nz = grid.num_rho, grid.num_theta, grid.num_zeta
        if not OMEGA_IS_0:
            coords = map_coordinates(
                eq,
                grid.nodes,
                ["rho", "theta", "phi"],
                ["rho", "theta", "zeta"],
                period=(np.inf, 2 * np.pi, 2 * np.pi),
                guess=grid.nodes,
            )
            grid = Grid(coords, sort=False)

    rows = np.floor(np.sqrt(nphi)).astype(int)
    cols = np.ceil(nphi / rows).astype(int)

    data, _ = _compute(
        eq,
        name,
        grid,
        kwargs.pop("component", None),
        reshape=False,
        compute_kwargs=compute_kwargs,
    )
    if normalize:
        norm_data, _ = _compute(
            eq,
            normalize,
            grid,
            reshape=False,
            compute_kwargs=compute_kwargs,
        )
        data = data / np.nanmean(np.abs(norm_data))  # normalize

    figw = 5 * cols
    figh = 5 * rows
    fig, ax = _format_ax(
        ax,
        rows=rows,
        cols=cols,
        figsize=kwargs.pop("figsize", (figw, figh)),
        equal=True,
    )
    ax = np.atleast_1d(ax).flatten()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coords = eq.compute(["R", "Z"], grid=grid)
    R = coords["R"].reshape((nt, nr, nz), order="F")
    Z = coords["Z"].reshape((nt, nr, nz), order="F")
    data = data.reshape((nt, nr, nz), order="F")
    op = "contour" + ("f" if kwargs.pop("fill", True) else "")
    contourf_kwargs = {}
    if log:
        data = np.abs(data)  # ensure data is positive for log plot
        contourf_kwargs["norm"] = matplotlib.colors.LogNorm()
        if name == "|F|_normalized" or (
            name == "|F|" and normalize == "<|grad(|B|^2)|/2mu0>_vol"
        ):
            contourf_kwargs["levels"] = kwargs.pop("levels", np.logspace(-6, 0, 7))
        else:
            logmin = max(np.floor(np.nanmin(np.log10(data))).astype(int), -16)
            logmax = np.ceil(np.nanmax(np.log10(data))).astype(int)
            contourf_kwargs["levels"] = kwargs.pop(
                "levels", np.logspace(logmin, logmax, logmax - logmin + 1)
            )
    else:
        contourf_kwargs["norm"] = matplotlib.colors.Normalize()
        contourf_kwargs["levels"] = kwargs.pop(
            "levels", np.linspace(data.min(), data.max(), 25)
        )
    contourf_kwargs["cmap"] = kwargs.pop("cmap", "jet")
    contourf_kwargs["extend"] = "both"
    title_fontsize = kwargs.pop("title_fontsize", None)
    xlabel_fontsize = kwargs.pop("xlabel_fontsize", None)
    ylabel_fontsize = kwargs.pop("ylabel_fontsize", None)
    cbar_label = kwargs.pop("cbar_label", None)

    data_index_p = data_index["desc.equilibrium.equilibrium.Equilibrium"]
    units = data_index_p[name]["units"]
    units = f"$(${units}$)" if (units != "~") else "$"

    suptitle = "$" + data_index_p[name]["label"] + units
    suptitle = kwargs.pop("suptitle", rf"{suptitle}")
    assert (
        len(kwargs) == 0
    ), f"plot section got unexpected keyword argument: {kwargs.keys()}"

    for i in range(nphi):
        cntr = getattr(ax[i], op)(
            R[:, :, i], Z[:, :, i], data[:, :, i], **contourf_kwargs
        )

        col = i % cols
        ax[i].tick_params(labelbottom=True, labelleft=(col == 0))
        ax[i].xaxis.set_minor_locator(AutoMinorLocator())
        for side in ["bottom", "left"]:
            ax[i].spines[side].set_linewidth(1.25)

        phi_title = "$\\phi /2\\pi = {:.3f}$".format(phi[i] / (2 * np.pi))
        ax[i].set_title(phi_title, fontsize=title_fontsize)
        if normalize:
            ax[i].set_title(
                "%s / %s, %s"
                % (
                    "$" + data_index_p[name]["label"] + "$",
                    "$" + data_index_p[normalize]["label"] + "$",
                    "$\\phi \\cdot N_{{FP}}/2\\pi = {:.3f}$".format(
                        eq.NFP * phi[i] / (2 * np.pi)
                    ),
                ),
                fontsize=title_fontsize,
            )

    if cbar_label is None:
        if normalize:
            cbar_label = "{} / {}".format(
                "$" + data_index_p[name]["label"] + "$",
                "$" + data_index_p[normalize]["label"] + "$",
            )
        else:
            cbar_label = r"$" + data_index_p[name]["label"] + units + "$"

    fig.suptitle(suptitle, fontsize=title_fontsize + 1)
    fig.supxlabel(_AXIS_LABELS_RPZ[0], fontsize=xlabel_fontsize)
    fig.supylabel(_AXIS_LABELS_RPZ[2], fontsize=ylabel_fontsize)
    fig.tight_layout(rect=[-0.03, -0.05, 0.9, 1.05])

    cbar_ax = fig.add_axes([0.9, 0.3, 0.015, 0.4])
    cbar = fig.colorbar(cntr, cax=cbar_ax, format="%.3f")
    cbar.update_ticks()
    cbar.set_label(cbar_label, fontsize=ylabel_fontsize)

    plot_data = {"R": R, "Z": Z, name: data}
    if normalize:
        plot_data["normalization"] = np.nanmean(np.abs(norm_data))
    else:
        plot_data["normalization"] = 1

    if return_data:
        return fig, ax, plot_data

    return fig, ax


def plot_surfaces(eq, rho=8, theta=8, phi=None, ax=None, return_data=False, **kwargs):
    """Plot flux surfaces.

    Parameters
    ----------
    eq : Equilibrium
        Object from which to plot.
    rho : int or array-like
        Values of rho to plot contours of.
        If an integer, plot that many contours linearly spaced in (0,1).
    theta : int or array-like
        Values of theta to plot contours of.
        If an integer, plot that many contours linearly spaced in (0,2pi).
    phi : float, int or array-like or None
        Values of phi to plot contours at.
        If an integer, plot that many contours linearly spaced in (0,2pi).
        Default is 1 contour for axisymmetric equilibria or 6 for non-axisymmetry.
    ax : matplotlib AxesSubplot, optional
        Axis to plot on.
    return_data : bool
        If True, return the data plotted as well as fig,ax
    **kwargs : dict, optional
        Specify properties of the figure, axis, and plot appearance e.g.::

            plot_X(figsize=(4,6),label="your_label")

        Valid keyword arguments are:

        * ``figsize``: tuple of length 2, the size of the figure (to be passed to
          matplotlib)
        * ``NR``: int, number of equispaced rho point to use in plotting the vartheta
          contours
        * ``NT``: int, number of equispaced theta points to use in plotting the rho
          contours
        * ``theta_color``: str or tuple, color to use for constant vartheta contours
        * ``theta_ls``: str, linestyle to use for constant vartheta contours
        * ``theta_lw``: float, linewidth to use for constant vartheta contours
        * ``rho_color``: str or tuple, color to use for constant rho contours
        * ``rho_ls``: str, linestyle to use for constant rho contours
        * ``rho_lw``: float, linewidth to use for constant rho contours
        * ``lcfs_color``: str or tuple, color to use for the LCFS constant rho contour
        * ``lcfs_ls``: str, linestyle to use for the LCFS constant rho contour
        * ``lcfs_lw``: float, linewidth to use for the LCFS constant rho contour
        * ``axis_color``: str or tuple, color to use for the axis plotted point
        * ``axis_alpha``: float, transparency of the axis plotted point
        * ``axis_marker``: str, markerstyle to use for the axis plotted point
        * ``axis_size``: float, markersize to use for the axis plotted point
        * ``title_fontsize``: integer, font size of the title
        * ``xlabel_fontsize``: float, fontsize of the xlabel
        * ``ylabel_fontsize``: float, fontsize of the ylabel
        * ``label``: str, label of the plotted line (e.g. to be shown with ax.legend())
        * ``legend``: bool, whether to show legend or not, False by default

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure being plotted to.
    ax : matplotlib.axes.Axes or ndarray of Axes
        Axes being plotted to.
    plot_data : dict
        Dictionary of the data plotted, only returned if ``return_data=True``

    Examples
    --------
    .. image:: ../../_static/images/plotting/plot_surfaces.png

    .. code-block:: python

        from desc.plotting import plot_surfaces
        fig, ax = plot_surfaces(eq)

    """
    phi = parse_argname_change(phi, kwargs, "zeta", "phi")

    NR = kwargs.pop("NR", 50)
    NT = kwargs.pop("NT", 180)
    figsize = kwargs.pop("figsize", None)
    theta_color = kwargs.pop("theta_color", colorblind_colors[2])
    theta_ls = kwargs.pop("theta_ls", ":")
    theta_lw = kwargs.pop("theta_lw", 1)
    rho_color = kwargs.pop("rho_color", colorblind_colors[0])
    rho_ls = kwargs.pop("rho_ls", "-")
    rho_lw = kwargs.pop("rho_lw", 1)
    lcfs_color = kwargs.pop("lcfs_color", colorblind_colors[1])
    lcfs_ls = kwargs.pop("lcfs_ls", "-")
    lcfs_lw = kwargs.pop("lcfs_lw", 1)
    axis_color = kwargs.pop("axis_color", colorblind_colors[3])
    axis_alpha = kwargs.pop("axis_alpha", 1)
    axis_marker = kwargs.pop("axis_marker", "o")
    axis_size = kwargs.pop("axis_size", 36)
    title_fontsize = kwargs.pop("title_fontsize", None)
    xlabel_fontsize = kwargs.pop("xlabel_fontsize", None)
    ylabel_fontsize = kwargs.pop("ylabel_fontsize", None)
    label = kwargs.pop("label", "")
    legend = kwargs.pop("legend", False if label == "" else True)

    assert (
        len(kwargs) == 0
    ), f"plot surfaces got unexpected keyword argument: {kwargs.keys()}"

    plot_theta = bool(theta)
    nfp = eq.NFP
    if isinstance(rho, numbers.Integral):
        rho = np.linspace(0, 1, rho + 1)
    rho = np.atleast_1d(rho)
    if isinstance(theta, numbers.Integral):
        theta = np.linspace(0, 2 * np.pi, theta, endpoint=False)
    theta = np.atleast_1d(theta)

    phi = (1 if eq.N == 0 else 6) if phi is None else phi
    if isinstance(phi, numbers.Integral):
        phi = np.linspace(0, 2 * np.pi / eq.NFP, phi, endpoint=False)
    phi = np.atleast_1d(phi)
    nphi = len(phi)

    # do not need NFP supplied to these grids as
    # the above logic takes care of the correct phi range
    # if defaults are requested. Setting NFP here instead
    # can create reshaping issues when phi is supplied and gets
    # truncated by 2pi/NFP. See PR #1204
    grid_kwargs = {
        "rho": rho,
        "NFP": 1,
        "theta": np.linspace(0, 2 * np.pi, NT, endpoint=True),
        "zeta": phi,
    }
    r_grid = _get_grid(**grid_kwargs)
    rnr, rnt, rnz = r_grid.num_rho, r_grid.num_theta, r_grid.num_zeta
    r_grid = Grid(
        map_coordinates(
            eq,
            r_grid.nodes,
            ["rho", "theta", "phi"],
            ["rho", "theta", "zeta"],
            period=(np.inf, 2 * np.pi, 2 * np.pi),
            guess=r_grid.nodes,
        ),
        sort=False,
    )
    grid_kwargs = {
        "rho": np.linspace(0, 1, NR),
        "NFP": 1,
        "theta": theta,
        "zeta": phi,
    }
    if plot_theta:
        # Note: theta* (also known as vartheta) is the poloidal straight field line
        # angle in PEST-like flux coordinates
        t_grid = _get_grid(**grid_kwargs)
        tnr, tnt, tnz = t_grid.num_rho, t_grid.num_theta, t_grid.num_zeta
        v_grid = Grid(
            map_coordinates(
                eq,
                t_grid.nodes,
                # TODO (#568): once generalized toroidal angle is used, change
                # inbasis to ["rho", "theta_PEST", "phi"],
                inbasis=["rho", "theta_PEST", "zeta"],
                outbasis=["rho", "theta", "zeta"],
                period=(np.inf, 2 * np.pi, 2 * np.pi),
                guess=t_grid.nodes,
                maxiter=30,
            ),
            sort=False,
        )
    rows = np.floor(np.sqrt(nphi)).astype(int)
    cols = np.ceil(nphi / rows).astype(int)

    # rho contours
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r_coords = eq.compute(["R", "Z"], grid=r_grid)
    Rr = r_coords["R"].reshape((rnt, rnr, rnz), order="F")
    Zr = r_coords["Z"].reshape((rnt, rnr, rnz), order="F")
    plot_data = {}

    if plot_theta:
        # vartheta contours
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            v_coords = eq.compute(["R", "Z"], grid=v_grid)
        Rv = v_coords["R"].reshape((tnt, tnr, tnz), order="F")
        Zv = v_coords["Z"].reshape((tnt, tnr, tnz), order="F")
        plot_data["vartheta_R_coords"] = Rv
        plot_data["vartheta_Z_coords"] = Zv

    figw = 4 * cols
    figh = 5 * rows
    if figsize is None:
        figsize = (figw, figh)
    fig, ax = _format_ax(ax, rows=rows, cols=cols, figsize=figsize, equal=True)
    ax = np.atleast_1d(ax).flatten()

    for i in range(nphi):
        if plot_theta:
            ax[i].plot(
                Rv[:, :, i].T,
                Zv[:, :, i].T,
                color=theta_color,
                linestyle=theta_ls,
                lw=theta_lw,
            )
        ax[i].plot(
            Rr[:, :, i], Zr[:, :, i], color=rho_color, linestyle=rho_ls, lw=rho_lw
        )
        ax[i].plot(
            Rr[:, -1, i],
            Zr[:, -1, i],
            color=lcfs_color,
            linestyle=lcfs_ls,
            lw=lcfs_lw,
            label=(label if i == 0 else ""),
        )
        if rho[0] == 0:
            ax[i].scatter(
                Rr[0, 0, i],
                Zr[0, 0, i],
                color=axis_color,
                alpha=axis_alpha,
                marker=axis_marker,
                s=axis_size,
            )

        if i == 0:
            ax[i].tick_params(labelbottom=True, labelleft=True)
        ax[i].xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax[i].yaxis.set_major_locator(MaxNLocator(nbins=7))

        ax[i].xaxis.set_minor_locator(AutoMinorLocator())
        ax[i].yaxis.set_minor_locator(AutoMinorLocator())

        ax[i].set_title(
            "$\\phi N_{{FP}} /2\\pi = {:.2f}$".format(nfp * phi[i] / (2 * np.pi)),
            fontsize=title_fontsize,
        )

        for side in ["bottom", "left"]:
            ax[i].spines[side].set_visible(True)
            ax[i].spines[side].set_linewidth(2)
        if label is not None and i == 0 and legend:
            ax[i].legend(loc="best")

    fig.supxlabel(_AXIS_LABELS_RPZ[0], fontsize=xlabel_fontsize)
    fig.supylabel(_AXIS_LABELS_RPZ[2], fontsize=ylabel_fontsize)
    _set_tight_layout(fig)

    plot_data["rho_R_coords"] = Rr
    plot_data["rho_Z_coords"] = Zr
    if return_data:
        return fig, ax, plot_data

    return fig, ax


def plot_comparison(
    eqs,
    rho=8,
    theta=8,
    phi=None,
    ax=None,
    cmap="rainbow",
    color=None,
    lw=None,
    ls=None,
    labels=None,
    return_data=False,
    **kwargs,
):
    """Plot comparison between flux surfaces of multiple equilibria.

    NOTE: If attempting to plot objects with differing NFP, `phi` must
    be given explicitly.

    Parameters
    ----------
    eqs : array-like of Equilibrium or EquilibriaFamily
        Equilibria to compare.
    rho : int or array-like
        Values of rho to plot contours of.
        If an integer, plot that many contours linearly spaced in (0,1).
    theta : int or array-like
        Values of theta to plot contours of.
        If an integer, plot that many contours linearly spaced in (0,2pi).
    phi : float, int or array-like or None
        Values of phi to plot contours at.
        If an integer, plot that many contours linearly spaced in [0,2pi).
        Default is 1 contour for axisymmetric equilibria or 6 for non-axisymmetry.
    ax : matplotlib AxesSubplot, optional
        Axis to plot on.
    cmap : str or matplotlib ColorMap
        Colormap to use for plotting, discretized into len(eqs) colors.
    color : array-like
        Array the same length as eqs of colors to use for each equilibrium.
        Overrides `cmap`.
    lw : array-like
        Array the same length as eqs of line widths to use for each equilibrium
    ls : array-like
        Array the same length as eqs of linestyles to use for each equilibrium.
    labels : array-like
        Array the same length as eqs of labels to apply to each equilibrium.
    return_data : bool
        If True, return the data plotted as well as fig,ax
    **kwargs : dict, optional
        Specify properties of the figure, axis, and plot appearance e.g.::

            plot_X(figsize=(4,6),label="your_label")

        Valid keyword arguments are:

        * ``figsize``: tuple of length 2, the size of the figure (to be passed to
          matplotlib)
        * ``legend``: bool, whether to display legend or not
        * ``legend_kw``: dict, any keyword arguments to be passed to ax.legend()
        * ``title_fontsize``: integer, font size of the title
        * ``xlabel_fontsize``: float, fontsize of the xlabel
        * ``ylabel_fontsize``: float, fontsize of the ylabel

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure being plotted to.
    ax : matplotlib.axes.Axes or ndarray of Axes
        Axes being plotted to.
    plot_data : dict
        Dictionary of the data plotted, only returned if ``return_data=True``

    Examples
    --------
    .. image:: ../../_static/images/plotting/plot_comparison.png

    .. code-block:: python

        from desc.plotting import plot_comparison
        fig, ax = plot_comparison(eqs=[eqf[0],eqf[1],eqf[2]],
                                  labels=['Axisymmetric w/o pressure',
                                          'Axisymmetric w/ pressure',
                                          'Non-axisymmetric w/ pressure',
                                         ],
                                 )

    """
    # if NFPs are not all equal, means there are
    # objects with differing NFPs, which it is not clear
    # how to choose the phis for by default, so we will throw an error
    # unless phi was given.
    phi = parse_argname_change(phi, kwargs, "zeta", "phi")
    errorif(
        not np.allclose([thing.NFP for thing in eqs], eqs[0].NFP) and phi is None,
        ValueError,
        "supplied objects must have the same number of field periods, "
        "or if there are differing field periods, `phi` must be given explicitly."
        f" Instead, supplied objects have NFPs {[t.NFP for t in eqs]}."
        " If attempting to plot an axisymmetric object with non-axisymmetric objects,"
        " you must use the `change_resolution` method to make the axisymmetric "
        "object have the same NFP as the non-axisymmetric objects.",
    )
    color = parse_argname_change(color, kwargs, "colors", "color")
    ls = parse_argname_change(ls, kwargs, "linestyles", "ls")
    lw = parse_argname_change(lw, kwargs, "lws", "lw")

    figsize = kwargs.pop("figsize", None)
    title_fontsize = kwargs.pop("title_fontsize", None)
    xlabel_fontsize = kwargs.pop("xlabel_fontsize", None)
    ylabel_fontsize = kwargs.pop("ylabel_fontsize", None)
    neq = len(eqs)
    if color is None:
        color = _get_cmap(cmap, neq)(np.linspace(0, 1, neq))
    if lw is None:
        lw = [1 for i in range(neq)]
    if ls is None:
        ls = ["-" for i in range(neq)]
    if labels is None:
        labels = [str(i) for i in range(neq)]
    N = np.max([eq.N for eq in eqs])
    nfp = eqs[0].NFP

    phi = (1 if N == 0 else 6) if phi is None else phi
    if isinstance(phi, numbers.Integral):
        phi = np.linspace(0, 2 * np.pi / nfp, phi, endpoint=False)
    phi = np.atleast_1d(phi)
    nphi = len(phi)

    rows = kwargs.pop("rows", np.floor(np.sqrt(nphi)).astype(int))
    cols = kwargs.pop("cols", np.ceil(nphi / rows).astype(int))

    figw = kwargs.pop("figw", 4 * cols - 1)
    figh = kwargs.pop("figh", 5 * rows)
    if figsize is None:
        figsize = (figw, figh)

    fig, ax = _format_ax(
        ax,
        rows=rows,
        cols=cols,
        figsize=figsize,
        equal=True,
    )
    ax = np.atleast_1d(ax).flatten()

    plot_data = {}
    for string in [
        "rho_R_coords",
        "rho_Z_coords",
        "vartheta_R_coords",
        "vartheta_Z_coords",
    ]:
        plot_data[string] = []
    for i, eq in enumerate(eqs):
        fig, ax, _plot_data = plot_surfaces(
            eq,
            rho,
            theta,
            phi,
            ax,
            theta_color=color[i % len(color)],
            theta_ls=ls[i % len(ls)],
            theta_lw=lw[i % len(lw)],
            rho_color=color[i % len(color)],
            rho_ls=ls[i % len(ls)],
            rho_lw=lw[i % len(lw)],
            lcfs_color=color[i % len(color)],
            lcfs_ls=ls[i % len(ls)],
            lcfs_lw=lw[i % len(lw)],
            axis_color=color[i % len(color)],
            axis_alpha=0,
            axis_marker="o",
            axis_size=0,
            label=labels[i % len(labels)],
            legend=False,
            title_fontsize=title_fontsize,
            xlabel_fontsize=xlabel_fontsize,
            ylabel_fontsize=ylabel_fontsize,
            return_data=True,
        )
        for key in _plot_data.keys():
            plot_data[key].append(_plot_data[key])

    if any(labels) and kwargs.pop("legend", True):
        fig.legend(**kwargs.pop("legend_kw", {}), frameon=False, fontsize=19)

    assert (
        len(kwargs) == 0
    ), f"plot_comparison got unexpected keyword argument: {kwargs.keys()}"

    if return_data:
        return fig, ax, plot_data

    return fig, ax
