"""Functions for plotting and visualizing equilibria."""

import inspect
import numbers
import tkinter
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib import cycler, rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable
from packaging.version import Version
from pylatexenc.latex2text import LatexNodes2Text
from termcolor import colored

from desc.backend import sign
from desc.basis import fourier, zernike_radial_poly
from desc.coils import CoilSet, _Coil
from desc.compute import data_index, get_transforms
from desc.compute.utils import _parse_parameterization
from desc.equilibrium.coords import map_coordinates
from desc.grid import Grid, LinearGrid
from desc.integrals import surface_averages_map
from desc.magnetic_fields import field_line_integrate
from desc.utils import errorif, islinspaced, only1, parse_argname_change, setdefault
from desc.vmec_utils import ptolemy_linear_transform

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
    "plot_grid",
    "plot_logo",
    "plot_qs_error",
    "plot_section",
    "plot_surfaces",
    "poincare_plot",
]


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
    # TODO: update this when matplotlib min version >= 3.6.0
    # compat layer to deal with API changes in mpl 3.6.0
    if Version(matplotlib.__version__) >= Version("3.6.0"):
        fig.set_layout_engine("tight")
    else:
        fig.set_tight_layout(True)


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
            fig = plt.figure(figsize=figsize, dpi=dpi)
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
        if isinstance(ax.flatten()[0], matplotlib.axes.Axes):
            return plt.gcf(), ax
        else:
            raise TypeError(
                colored(
                    "ax argument must be None or an axis instance or array of axes",
                    "red",
                )
            )


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


def _compute(eq, name, grid, component=None, reshape=True):
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

    Returns
    -------
    data : float array of shape (M, L, N)
        Computed quantity.

    """
    parameterization = _parse_parameterization(eq)
    if name not in data_index[parameterization]:
        raise ValueError(
            f"Unrecognized value '{name}' for "
            + f"parameterization {parameterization}."
        )
    assert component in [
        None,
        "R",
        "phi",
        "Z",
    ], f"component must be one of [None, 'R', 'phi', 'Z'], got {component}"

    components = {"R": 0, "phi": 1, "Z": 2}

    label = data_index[parameterization][name]["label"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = eq.compute(name, grid=grid)[name]

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

    label = r"$" + label + "~(" + data_index[parameterization][name]["units"] + ")$"

    if reshape:
        data = data.reshape((grid.num_theta, grid.num_rho, grid.num_zeta), order="F")

    return data, label


def _compute_Bn(eq, field, plot_grid, field_grid):
    """Compute normal field from virtual casing + coils, using correct grids."""
    errorif(
        field is None,
        ValueError,
        "If B*n is entered as the variable to plot, a magnetic field"
        " must be provided.",
    )
    errorif(
        not np.all(np.isclose(plot_grid.nodes[:, 0], 1)),
        ValueError,
        "If B*n is entered as the variable to plot, "
        "the grid nodes must be at rho=1.",
    )

    theta_endpoint = zeta_endpoint = False

    if plot_grid.fft_poloidal and plot_grid.fft_toroidal:
        source_grid = eval_grid = plot_grid
    # often plot_grid is still linearly spaced but includes endpoints. In that case
    # make a temp grid that just leaves out the endpoint so we can FFT
    elif (
        isinstance(plot_grid, LinearGrid)
        and not plot_grid.sym
        and islinspaced(plot_grid.nodes[plot_grid.unique_theta_idx, 1])
        and islinspaced(plot_grid.nodes[plot_grid.unique_zeta_idx, 2])
    ):
        if plot_grid._poloidal_endpoint:
            theta_endpoint = True
            theta = plot_grid.nodes[plot_grid.unique_theta_idx[0:-1], 1]
        if plot_grid._toroidal_endpoint:
            zeta_endpoint = True
            zeta = plot_grid.nodes[plot_grid.unique_zeta_idx[0:-1], 2]
        vc_grid = LinearGrid(
            theta=theta,
            zeta=zeta,
            NFP=plot_grid.NFP,
            endpoint=False,
        )
        # override attr since we know fft is ok even with custom nodes
        vc_grid._fft_poloidal = vc_grid._fft_toroidal = True
        source_grid = eval_grid = vc_grid
    else:
        eval_grid = plot_grid
        source_grid = LinearGrid(
            M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=False, endpoint=False
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        data, _ = field.compute_Bnormal(
            eq, eval_grid=eval_grid, source_grid=field_grid, vc_source_grid=source_grid
        )
    data = data.reshape((eval_grid.num_theta, eval_grid.num_zeta), order="F")
    if theta_endpoint:
        data = np.vstack((data, data[0, :]))
    if zeta_endpoint:
        data = np.hstack((data, np.atleast_2d(data[:, 0]).T))
    data = data.reshape(
        (plot_grid.num_theta, plot_grid.num_rho, plot_grid.num_zeta), order="F"
    )

    label = r"$\mathbf{B} \cdot \hat{n} ~(\mathrm{T})$"
    return data, label


def plot_coefficients(eq, L=True, M=True, N=True, ax=None, **kwargs):
    """Plot spectral coefficient magnitudes vs spectral mode number.

    Parameters
    ----------
    eq : Equilibrium
        Object from which to plot.
    L : bool
        Whether to include radial mode numbers in the x-axis or not.
    M : bool
        Whether to include poloidal mode numbers in the x-axis or not.
    N : bool
        Whether to include toroidal mode numbers in the x-axis or not.
    ax : matplotlib AxesSubplot, optional
        Axis to plot on.
    **kwargs : fig,ax and plotting properties
        Specify properties of the figure, axis, and plot appearance e.g.::

            plot_X(figsize=(4,6))

        Valid keyword arguments are:

        figsize: tuple of length 2, the size of the figure (to be passed to matplotlib)
        title_fontsize: integer, font size of the title
        xlabel_fontsize: integer, font size of the x axis label
        color: str or tuple, color to use for scatter plot
        marker: str, marker to use for scatter plot


    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure being plotted to.
    ax : matplotlib.axes.Axes or ndarray of Axes
        Axes being plotted to.

    Examples
    --------
    .. image:: ../../_static/images/plotting/plot_coefficients.png

    .. code-block:: python

        from desc.plotting import plot_coefficients
        fig, ax = plot_coefficients(eq)

    """
    lmn = np.array([], dtype=int)
    xlabel = ""
    if L:
        lmn = np.append(lmn, np.array([0]))
        xlabel += "l"
        if M or N:
            xlabel += " + "
    if M:
        lmn = np.append(lmn, np.array([1]))
        xlabel += "|m|"
        if N:
            xlabel += " + "
    if N:
        lmn = np.append(lmn, np.array([2]))
        xlabel += "|n|"

    fig, ax = _format_ax(ax, rows=1, cols=3, figsize=kwargs.pop("figsize", None))
    title_fontsize = kwargs.pop("title_fontsize", None)
    xlabel_fontsize = kwargs.pop("xlabel_fontsize", None)
    marker = kwargs.pop("marker", "o")
    color = kwargs.pop("color", "b")

    assert (
        len(kwargs) == 0
    ), f"plot_coefficients got unexpected keyword argument: {kwargs.keys()}"

    ax[0, 0].semilogy(
        np.sum(np.abs(eq.R_basis.modes[:, lmn]), axis=1),
        np.abs(eq.R_lmn),
        c=color,
        marker=marker,
        ls="",
    )
    ax[0, 1].semilogy(
        np.sum(np.abs(eq.Z_basis.modes[:, lmn]), axis=1),
        np.abs(eq.Z_lmn),
        c=color,
        marker=marker,
        ls="",
    )
    ax[0, 2].semilogy(
        np.sum(np.abs(eq.L_basis.modes[:, lmn]), axis=1),
        np.abs(eq.L_lmn),
        c=color,
        marker=marker,
        ls="",
    )

    ax[0, 0].set_xlabel(xlabel, fontsize=xlabel_fontsize)
    ax[0, 1].set_xlabel(xlabel, fontsize=xlabel_fontsize)
    ax[0, 2].set_xlabel(xlabel, fontsize=xlabel_fontsize)

    ax[0, 0].set_title("$|R_{lmn}|$", fontsize=title_fontsize)
    ax[0, 1].set_title("$|Z_{lmn}|$", fontsize=title_fontsize)
    ax[0, 2].set_title("$|\\lambda_{lmn}|$", fontsize=title_fontsize)
    _set_tight_layout(fig)

    return fig, ax


def plot_1d(eq, name, grid=None, log=False, ax=None, return_data=False, **kwargs):
    """Plot 1D profiles.

    Parameters
    ----------
    eq : Equilibrium, Surface, Curve
        Object from which to plot.
    name : str
        Name of variable to plot.
    grid : Grid, optional
        Grid of coordinates to plot at.
    log : bool, optional
        Whether to use a log scale.
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
        * ``component``: str, one of [None, 'R', 'phi', 'Z'], For vector variables,
          which element to plot. Default is the norm of the vector.
        * ``label``: str, label of the plotted line (e.g. to be shown with ax.legend())
        * ``xlabel_fontsize``: float, fontsize of the xlabel
        * ``ylabel_fontsize``: float, fontsize of the ylabel
        * ``linecolor``: str or tuple, color to use for plot line
        * ``ls``: str, linestyle to use for plot line
        * ``lw``: float, linewidth to use for plot line

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
    .. image:: ../../_static/images/plotting/plot_1d.png

    .. code-block:: python

        from desc.plotting import plot_1d
        plot_1d(eq, 'p')

    """
    # If the quantity is a flux surface function, call plot_fsa.
    # This is done because the computation of some quantities relies on a
    # surface average. Surface averages should be computed over a 2-D grid to
    # sample the entire surface. Computing this on a 1-D grid would return a
    # misleading plot.
    parameterization = _parse_parameterization(eq)
    default_L = 100
    default_N = 0
    if data_index[parameterization][name]["coordinates"] == "r":
        if grid is None:
            return plot_fsa(
                eq,
                name,
                rho=default_L,
                log=log,
                ax=ax,
                return_data=return_data,
                grid=grid,
                **kwargs,
            )
        rho = grid.nodes[:, 0]
        if not np.all(np.isclose(rho, rho[0])):
            # rho nodes are not constant, so user must be plotting against rho
            return plot_fsa(
                eq,
                name,
                rho=rho,
                log=log,
                ax=ax,
                return_data=return_data,
                grid=grid,
                **kwargs,
            )

    elif data_index[parameterization][name]["coordinates"] == "s":  # curve qtys
        default_L = 0
        default_N = 100
    NFP = getattr(eq, "NFP", 1)
    if grid is None:
        grid_kwargs = {"L": default_L, "N": default_N, "NFP": NFP}
        grid = _get_grid(**grid_kwargs)
    plot_axes = _get_plot_axes(grid)
    if len(plot_axes) != 1:
        return ValueError(colored("Grid must be 1D", "red"))

    data, ylabel = _compute(eq, name, grid, kwargs.pop("component", None))
    label = kwargs.pop("label", None)

    fig, ax = _format_ax(ax, figsize=kwargs.pop("figsize", None))

    # reshape data to 1D
    data = data.flatten()
    linecolor = kwargs.pop("linecolor", colorblind_colors[0])
    ls = kwargs.pop("ls", "-")
    lw = kwargs.pop("lw", 1)
    if log:
        data = np.abs(data)  # ensure data is positive for log plot
        ax.semilogy(
            grid.nodes[:, plot_axes[0]],
            data,
            label=label,
            color=linecolor,
            ls=ls,
            lw=lw,
        )
    else:
        ax.plot(
            grid.nodes[:, plot_axes[0]],
            data,
            label=label,
            color=linecolor,
            ls=ls,
            lw=lw,
        )
    xlabel_fontsize = kwargs.pop("xlabel_fontsize", None)
    ylabel_fontsize = kwargs.pop("ylabel_fontsize", None)

    assert len(kwargs) == 0, f"plot_1d got unexpected keyword argument: {kwargs.keys()}"
    xlabel = _AXIS_LABELS_RTZ[plot_axes[0]]
    ax.set_xlabel(xlabel, fontsize=xlabel_fontsize)
    ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
    _set_tight_layout(fig)
    plot_data = {xlabel.strip("$").strip("\\"): grid.nodes[:, plot_axes[0]], name: data}

    if label is not None:
        ax.legend()

    if return_data:
        return fig, ax, plot_data

    return fig, ax


def plot_2d(
    eq, name, grid=None, log=False, norm_F=False, ax=None, return_data=False, **kwargs
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
    norm_F : bool, optional
        Whether to normalize a plot of force error to be unitless.
        Vacuum equilibria are normalized by the gradient of magnetic pressure,
        while finite beta equilibria are normalized by the pressure gradient.
    ax : matplotlib AxesSubplot, optional
        Axis to plot on.
    return_data : bool
        If True, return the data plotted as well as fig,ax
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
        * ``levels``: int or array-like, passed to contourf
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
    parameterization = _parse_parameterization(eq)
    if grid is None:
        grid_kwargs = {"M": 33, "N": 33, "NFP": eq.NFP, "axis": False}
        grid = _get_grid(**grid_kwargs)
    plot_axes = _get_plot_axes(grid)
    if len(plot_axes) != 2:
        return ValueError(colored("Grid must be 2D", "red"))
    component = kwargs.pop("component", None)
    if name != "B*n":
        data, label = _compute(
            eq,
            name,
            grid,
            component=component,
        )
    else:
        data, label = _compute_Bn(
            eq, kwargs.pop("field", None), grid, kwargs.pop("field_grid", None)
        )

    fig, ax = _format_ax(ax, figsize=kwargs.pop("figsize", None))
    divider = make_axes_locatable(ax)

    if norm_F:
        # normalize force by B pressure gradient
        norm_name = kwargs.pop("norm_name", "<|grad(|B|^2)|/2mu0>_vol")
        norm_data, _ = _compute(eq, norm_name, grid, reshape=False)
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
        if norm_F:
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
            "levels", np.linspace(np.nanmin(data), np.nanmax(data), 100)
        )
    contourf_kwargs["cmap"] = kwargs.pop("cmap", "jet")
    contourf_kwargs["extend"] = "both"
    title_fontsize = kwargs.pop("title_fontsize", None)
    xlabel_fontsize = kwargs.pop("xlabel_fontsize", None)
    ylabel_fontsize = kwargs.pop("ylabel_fontsize", None)
    filled = kwargs.pop("filled", True)
    assert len(kwargs) == 0, f"plot_2d got unexpected keyword argument: {kwargs.keys()}"

    cax_kwargs = {"size": "5%", "pad": 0.05}

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
    cax = divider.append_axes("right", **cax_kwargs)
    cbar = fig.colorbar(im, cax=cax)
    cbar.update_ticks()
    xlabel = _AXIS_LABELS_RTZ[plot_axes[1]]
    ylabel = _AXIS_LABELS_RTZ[plot_axes[0]]
    ax.set_xlabel(xlabel, fontsize=xlabel_fontsize)
    ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
    ax.set_title(label, fontsize=title_fontsize)
    if norm_F:
        ax.set_title(
            "%s / %s"
            % (
                "$" + data_index[parameterization][name]["label"] + "$",
                "$" + data_index[parameterization][norm_name]["label"] + "$",
            )
        )
    _set_tight_layout(fig)
    plot_data = {
        xlabel.strip("$").strip("\\"): xx,
        ylabel.strip("$").strip("\\"): yy,
        name: data,
    }

    if norm_F:
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


def plot_3d(
    eq,
    name,
    grid=None,
    log=False,
    fig=None,
    return_data=False,
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
    fig : plotly.graph_objs._figure.Figure, optional
        Figure to plot on.
    return_data : bool
        If True, return the data plotted as well as fig,ax
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
    .. image:: ../../_static/images/plotting/plot_3d.png

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

    if name != "B*n":
        data, label = _compute(
            eq,
            name,
            grid,
            component=component,
        )
    else:
        data, label = _compute_Bn(
            eq, kwargs.pop("field", None), grid, kwargs.pop("field_grid", None)
        )

    errorif(
        len(kwargs) != 0,
        ValueError,
        f"plot_3d got unexpected keyword argument: {kwargs.keys()}",
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
        font=dict(family="Times"),
    )
    plot_data = {"X": X, "Y": Y, "Z": Z, name: data}

    if return_data:
        return fig, plot_data

    return fig


def plot_fsa(  # noqa: C901
    eq,
    name,
    with_sqrt_g=True,
    log=False,
    rho=20,
    M=None,
    N=None,
    norm_F=False,
    ax=None,
    return_data=False,
    grid=None,
    **kwargs,
):
    """Plot flux surface averages of quantities.

    Parameters
    ----------
    eq : Equilibrium
        Object from which to plot.
    name : str
        Name of variable to plot.
    with_sqrt_g : bool, optional
        Whether to weight the surface average with sqrt(g), the 3-D Jacobian
        determinant of flux coordinate system. Default is True.

        The weighted surface average is also known as a flux surface average.
        The unweighted surface average is also known as a theta average.

        Note that this boolean has no effect for quantities which are defined
        as surface functions because averaging such functions is the identity
        operation.
    log : bool, optional
        Whether to use a log scale.
    rho : int or array-like
        Values of rho to plot contours of.
        If an integer, plot that many contours linearly spaced in (0,1).
    M : int, optional
        Poloidal grid resolution. Default is eq.M_grid.
    N : int, optional
        Toroidal grid resolution. Default is eq.N_grid.
    norm_F : bool, optional
        Whether to normalize a plot of force error to be unitless.
        Vacuum equilibria are normalized by the volume average of the gradient
        of magnetic pressure, while finite beta equilibria are normalized by the
        volume average of the pressure gradient.
    ax : matplotlib AxesSubplot, optional
        Axis to plot on.
    return_data : bool
        if True, return the data plotted as well as fig,ax
    grid : _Grid
        Grid to compute name on. If provided, the parameters
        ``rho``, ``M``, and ``N`` are ignored.
    **kwargs : dict, optional
        Specify properties of the figure, axis, and plot appearance e.g.::

            plot_X(figsize=(4,6),label="your_label")

        Valid keyword arguments are:

        * ``figsize``: tuple of length 2, the size of the figure (to be passed to
          matplotlib)
        * ``component``: str, one of [None, 'R', 'phi', 'Z'], For vector variables,
          which element to plot. Default is the norm of the vector.
        * ``label``: str, label of the plotted line (e.g. to be shown with ax.legend())
        * ``xlabel_fontsize``: float, fontsize of the xlabel
        * ``ylabel_fontsize``: float, fontsize of the ylabel
        * ``linecolor``: str or tuple, color to use for plot line
        * ``ls``: str, linestyle to use for plot line
        * ``lw``: float, linewidth to use for plot line

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
    .. image:: ../../_static/images/plotting/plot_fsa.png

    .. code-block:: python

        from desc.plotting import plot_fsa
        fig, ax = plot_fsa(eq, "B_theta", with_sqrt_g=False)

    """
    if M is None:
        M = eq.M_grid
    if N is None:
        N = eq.N_grid
    if grid is None:
        if np.isscalar(rho) and (int(rho) == rho):
            rho = np.linspace(0, 1, rho + 1)
        rho = np.atleast_1d(rho)
        grid = LinearGrid(M=M, N=N, NFP=eq.NFP, sym=eq.sym, rho=rho)
    else:
        rho = grid.compress(grid.nodes[:, 0])

    linecolor = kwargs.pop("linecolor", colorblind_colors[0])
    ls = kwargs.pop("ls", "-")
    lw = kwargs.pop("lw", 1)
    fig, ax = _format_ax(ax, figsize=kwargs.pop("figsize", (4, 4)))

    label = kwargs.pop("label", None)
    p = "desc.equilibrium.equilibrium.Equilibrium"
    if "<" + name + ">" in data_index[p]:
        # If we identify the quantity to plot as something in data_index, then
        # we may be able to compute more involved magnetic axis limits.
        deps = data_index[p]["<" + name + ">"]["dependencies"]["data"]
        if with_sqrt_g == ("sqrt(g)" in deps or "V_r(r)" in deps):
            # When we denote a quantity as ``<name>`` in data_index, we have
            # marked it a surface average of ``name``. This does not specify
            # the type of surface average however (i.e. with/without the sqrt(g)
            # factor). The above conditional guard should ensure that the
            # surface average we have the recipe to compute in data_index is the
            # desired surface average.
            name = "<" + name + ">"
    values, ylabel = _compute(
        eq, name, grid, kwargs.pop("component", None), reshape=False
    )
    ylabel = ylabel.split("~")
    if (
        data_index[p][name]["coordinates"] == "r"
        or data_index[p][name]["coordinates"] == ""
    ):
        # If the quantity is a surface function, averaging it again has no
        # effect, regardless of whether sqrt(g) is used.
        # So we avoid surface averaging it and forgo the <> around the ylabel.
        ylabel = r"$ " + ylabel[0][1:] + r" ~" + "~".join(ylabel[1:])
        plot_data_ylabel_key = f"{name}"
        if data_index[p][name]["coordinates"] == "r":
            values = grid.compress(values)
    else:
        compute_surface_averages = surface_averages_map(grid, expand_out=False)
        if with_sqrt_g:  # flux surface average
            sqrt_g = _compute(eq, "sqrt(g)", grid, reshape=False)[0]
            # Attempt to compute the magnetic axis limit.
            # Compute derivative depending on various naming schemes.
            # e.g. B -> B_r, V(r) -> V_r(r), S_r(r) -> S_rr(r)
            # psi_r/sqrt(g) -> (psi_r/sqrt(g))_r
            schemes = (
                name + "_r",
                name[:-3] + "_r" + name[-3:],
                name[:-3] + "r" + name[-3:],
                "(" + name + ")_r",
            )
            values_r = next(
                (
                    _compute(eq, x, grid, reshape=False)[0]
                    for x in schemes
                    if x in data_index[p]
                ),
                np.nan,
            )
            if (np.isfinite(values) & np.isfinite(values_r))[grid.axis].all():
                # Otherwise cannot compute axis limit in this agnostic manner.
                sqrt_g = grid.replace_at_axis(
                    sqrt_g, _compute(eq, "sqrt(g)_r", grid, reshape=False)[0], copy=True
                )
            averages = compute_surface_averages(values, sqrt_g=sqrt_g)
            ylabel = r"$\langle " + ylabel[0][1:] + r" \rangle~" + "~".join(ylabel[1:])
        else:  # theta average
            averages = compute_surface_averages(values)
            ylabel = (
                r"$\langle "
                + ylabel[0][1:]
                + r" \rangle_{\theta}~"
                + "~".join(ylabel[1:])
            )
        # True if values has nan on a given surface.
        is_nan = compute_surface_averages(np.isnan(values)).astype(bool)
        # The integration replaced nan with 0.
        # Put them back to avoid misleading plot (e.g. cusp near the magnetic axis).
        values = np.where(is_nan, np.nan, averages)
        plot_data_ylabel_key = f"<{name}>_fsa"

    if norm_F:
        # normalize force by B pressure gradient
        norm_name = kwargs.pop("norm_name", "<|grad(|B|^2)|/2mu0>_vol")
        norm_data = _compute(eq, norm_name, grid, reshape=False)[0]
        values = values / np.nanmean(np.abs(norm_data))  # normalize
    if log:
        values = np.abs(values)  # ensure data is positive for log plot
        ax.semilogy(rho, values, label=label, color=linecolor, ls=ls, lw=lw)
    else:
        ax.plot(rho, values, label=label, color=linecolor, ls=ls, lw=lw)
    xlabel_fontsize = kwargs.pop("xlabel_fontsize", None)
    ylabel_fontsize = kwargs.pop("ylabel_fontsize", None)
    assert (
        len(kwargs) == 0
    ), f"plot_fsa got unexpected keyword argument: {kwargs.keys()}"

    ax.set_xlabel(_AXIS_LABELS_RTZ[0], fontsize=xlabel_fontsize)
    ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
    if norm_F:
        ax.set_ylabel(
            "%s / %s"
            % (
                "$" + data_index[p][name]["label"] + "$",
                "$" + data_index[p][norm_name]["label"] + "$",
            ),
            fontsize=ylabel_fontsize,
        )
    _set_tight_layout(fig)

    if label is not None:
        ax.legend()

    plot_data = {"rho": rho, plot_data_ylabel_key: values}
    if norm_F:
        plot_data["normalization"] = np.nanmean(np.abs(norm_data))
    else:
        plot_data["normalization"] = 1

    if return_data:
        return fig, ax, plot_data

    return fig, ax


def plot_section(
    eq, name, grid=None, log=False, norm_F=False, ax=None, return_data=False, **kwargs
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
    norm_F : bool, optional
        Whether to normalize a plot of force error to be unitless.
        Vacuum equilibria are normalized by the gradient of magnetic pressure,
        while finite beta equilibria are normalized by the pressure gradient.
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
        * ``component``: str, one of [None, 'R', 'phi', 'Z'], For vector variables,
          which element to plot. Default is the norm of the vector.
        * ``title_fontsize``: integer, font size of the title
        * ``xlabel_fontsize``: float, fontsize of the xlabel
        * ``ylabel_fontsize``: float, fontsize of the ylabel
        * ``cmap``: str, matplotlib colormap scheme to use, passed to ax.contourf
        * ``levels``: int or array-like, passed to contourf
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
    phi = kwargs.pop("phi", (1 if eq.N == 0 else 6))
    phi = parse_argname_change(phi, kwargs, "nzeta", "phi")
    phi = parse_argname_change(phi, kwargs, "nphi", "phi")

    if isinstance(phi, numbers.Integral):
        phi = np.linspace(0, 2 * np.pi / eq.NFP, phi, endpoint=False)
    phi = np.atleast_1d(phi)
    nphi = len(phi)
    if grid is None:
        grid_kwargs = {
            "L": 25,
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

    data, label = _compute(eq, name, grid, kwargs.pop("component", None), reshape=False)
    if norm_F:
        # normalize force by B pressure gradient
        norm_name = kwargs.pop("norm_name", "<|grad(|B|^2)|/2mu0>_vol")
        norm_data, _ = _compute(eq, norm_name, grid, reshape=False)
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
        if norm_F:
            contourf_kwargs["levels"] = kwargs.pop("levels", np.logspace(-6, 0, 7))
        else:
            logmin = np.floor(np.nanmin(np.log10(data))).astype(int)
            logmax = np.ceil(np.nanmax(np.log10(data))).astype(int)
            contourf_kwargs["levels"] = kwargs.pop(
                "levels", np.logspace(logmin, logmax, logmax - logmin + 1)
            )
    else:
        contourf_kwargs["norm"] = matplotlib.colors.Normalize()
        contourf_kwargs["levels"] = kwargs.pop(
            "levels", np.linspace(data.min(), data.max(), 100)
        )
    contourf_kwargs["cmap"] = kwargs.pop("cmap", "jet")
    contourf_kwargs["extend"] = "both"
    title_fontsize = kwargs.pop("title_fontsize", None)
    xlabel_fontsize = kwargs.pop("xlabel_fontsize", None)
    ylabel_fontsize = kwargs.pop("ylabel_fontsize", None)
    assert (
        len(kwargs) == 0
    ), f"plot section got unexpected keyword argument: {kwargs.keys()}"

    cax_kwargs = {"size": "5%", "pad": 0.05}

    for i in range(nphi):
        divider = make_axes_locatable(ax[i])

        cntr = getattr(ax[i], op)(
            R[:, :, i], Z[:, :, i], data[:, :, i], **contourf_kwargs
        )
        cax = divider.append_axes("right", **cax_kwargs)
        cbar = fig.colorbar(cntr, cax=cax)
        cbar.update_ticks()

        ax[i].set_xlabel(_AXIS_LABELS_RPZ[0], fontsize=xlabel_fontsize)
        ax[i].set_ylabel(_AXIS_LABELS_RPZ[2], fontsize=ylabel_fontsize)
        ax[i].tick_params(labelbottom=True, labelleft=True)
        ax[i].set_title(
            "$"
            + data_index["desc.equilibrium.equilibrium.Equilibrium"][name]["label"]
            + "$ ($"
            + data_index["desc.equilibrium.equilibrium.Equilibrium"][name]["units"]
            + "$)"
            + ", $\\phi \\cdot N_{{FP}}/2\\pi = {:.3f}$".format(
                eq.NFP * phi[i] / (2 * np.pi)
            )
        )
        if norm_F:
            ax[i].set_title(
                "%s / %s, %s"
                % (
                    "$"
                    + data_index["desc.equilibrium.equilibrium.Equilibrium"][name][
                        "label"
                    ]
                    + "$",
                    "$"
                    + data_index["desc.equilibrium.equilibrium.Equilibrium"][norm_name][
                        "label"
                    ]
                    + "$",
                    "$\\phi \\cdot N_{{FP}}/2\\pi = {:.3f}$".format(
                        eq.NFP * phi[i] / (2 * np.pi)
                    ),
                ),
                fontsize=title_fontsize,
            )
    _set_tight_layout(fig)

    plot_data = {"R": R, "Z": Z, name: data}
    if norm_F:
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
        * ``label``: str, label of the plotted line (e.g. to be shown with ax.legend())
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
    label = kwargs.pop("label", "")
    title_fontsize = kwargs.pop("title_fontsize", None)
    xlabel_fontsize = kwargs.pop("xlabel_fontsize", None)
    ylabel_fontsize = kwargs.pop("ylabel_fontsize", None)

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
                ["rho", "theta_PEST", "phi"],
                ["rho", "theta", "zeta"],
                period=(np.inf, 2 * np.pi, 2 * np.pi),
                guess=t_grid.nodes,
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

        ax[i].set_xlabel(_AXIS_LABELS_RPZ[0], fontsize=xlabel_fontsize)
        ax[i].set_ylabel(_AXIS_LABELS_RPZ[2], fontsize=ylabel_fontsize)
        ax[i].tick_params(labelbottom=True, labelleft=True)
        ax[i].set_title(
            "$\\phi \\cdot N_{{FP}}/2\\pi = {:.3f}$".format(nfp * phi[i] / (2 * np.pi)),
            fontsize=title_fontsize,
        )
        if label is not None and i == 0 and kwargs.pop("legend", False):
            ax[i].legend()

    _set_tight_layout(fig)

    plot_data["rho_R_coords"] = Rr
    plot_data["rho_Z_coords"] = Zr
    if return_data:
        return fig, ax, plot_data

    return fig, ax


def poincare_plot(
    field,
    R0,
    Z0,
    ntransit=100,
    phi=None,
    NFP=None,
    grid=None,
    ax=None,
    return_data=False,
    **kwargs,
):
    """Poincare plot of field lines from external magnetic field.

    Parameters
    ----------
    field : MagneticField
        External field, coilset, current potential etc to plot from.
    R0, Z0 : array-like
        Starting points at phi=0 for field line tracing.
    ntransit : int
        Number of transits to trace field lines for.
    phi : float, int or array-like or None
        Values of phi to plot section at.
        If an integer, plot that many contours linearly spaced in (0,2pi).
        Default is 6.
    NFP : int, optional
        Number of field periods. By default attempts to infer from ``field``, otherwise
        uses NFP=1.
    grid : Grid, optional
        Grid used to discretize ``field``.
    ax : matplotlib AxesSubplot, optional
        Axis to plot on.
    return_data : bool
        If True, return the data plotted as well as fig,ax
    **kwargs : dict, optional
        Specify properties of the figure, axis, and plot appearance e.g.::

            plot_X(figsize=(4,6),)

        Valid keyword arguments are:

        * ``figsize``: tuple of length 2, the size of the figure (to be passed to
          matplotlib)
        * ``color``: str or tuple, color to use for field lines.
        * ``marker``: str, markerstyle to use for the plotted points
        * ``size``: float, markersize to use for the plotted points
        * ``title_fontsize``: integer, font size of the title
        * ``xlabel_fontsize``: float, fontsize of the xlabel
        * ``ylabel_fontsize``: float, fontsize of the ylabel

        Additionally, any other keyword arguments will be passed on to
        ``desc.magnetic_fields.field_line_integrate``

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure being plotted to.
    ax : matplotlib.axes.Axes or ndarray of Axes
        Axes being plotted to.
    plot_data : dict
        Dictionary of the data plotted, only returned if ``return_data=True``
    """
    fli_kwargs = {}
    for key in inspect.signature(field_line_integrate).parameters:
        if key in kwargs:
            fli_kwargs[key] = kwargs.pop(key)

    figsize = kwargs.pop("figsize", None)
    color = kwargs.pop("color", colorblind_colors[0])
    marker = kwargs.pop("marker", "o")
    size = kwargs.pop("size", 5)
    title_fontsize = kwargs.pop("title_fontsize", None)
    xlabel_fontsize = kwargs.pop("xlabel_fontsize", None)
    ylabel_fontsize = kwargs.pop("ylabel_fontsize", None)

    assert (
        len(kwargs) == 0
    ), f"poincare_plot got unexpected keyword argument: {kwargs.keys()}"

    if NFP is None:
        NFP = getattr(field, "NFP", 1)

    phi = 6 if phi is None else phi
    if isinstance(phi, numbers.Integral):
        phi = np.linspace(0, 2 * np.pi / NFP, phi, endpoint=False)
    phi = np.atleast_1d(phi)
    nplanes = len(phi)

    phis = (phi + np.arange(0, ntransit)[:, None] * 2 * np.pi / NFP).flatten()

    R0, Z0 = np.atleast_1d(R0, Z0)

    fieldR, fieldZ = field_line_integrate(
        r0=R0,
        z0=Z0,
        phis=phis,
        field=field,
        source_grid=grid,
        **fli_kwargs,
    )

    zs = fieldZ.reshape((ntransit, nplanes, -1))
    rs = fieldR.reshape((ntransit, nplanes, -1))

    signBT = np.sign(
        field.compute_magnetic_field(np.array([R0.flat[0], 0.0, Z0.flat[0]]))[:, 1]
    ).flat[0]
    if signBT < 0:  # field lines are traced backwards when toroidal field < 0
        rs, zs = rs[:, ::-1], zs[:, ::-1]
        rs, zs = np.roll(rs, 1, 1), np.roll(zs, 1, 1)

    data = {
        "R": rs,
        "Z": zs,
    }

    rows = np.floor(np.sqrt(nplanes)).astype(int)
    cols = np.ceil(nplanes / rows).astype(int)

    figw = 4 * cols
    figh = 5 * rows
    if figsize is None:
        figsize = (figw, figh)
    fig, ax = _format_ax(ax, rows=rows, cols=cols, figsize=figsize, equal=True)

    for i in range(nplanes):
        ax.flat[i].scatter(
            rs[:, i, :],
            zs[:, i, :],
            color=color,
            marker=marker,
            s=size,
        )

        ax.flat[i].set_xlabel(_AXIS_LABELS_RPZ[0], fontsize=xlabel_fontsize)
        ax.flat[i].set_ylabel(_AXIS_LABELS_RPZ[2], fontsize=ylabel_fontsize)
        ax.flat[i].tick_params(labelbottom=True, labelleft=True)
        ax.flat[i].set_title(
            "$\\phi \\cdot N_{{FP}}/2\\pi = {:.3f}$".format(NFP * phi[i] / (2 * np.pi)),
            fontsize=title_fontsize,
        )

    _set_tight_layout(fig)

    if return_data:
        return fig, ax, data
    return fig, ax


def plot_boundary(eq, phi=None, plot_axis=True, ax=None, return_data=False, **kwargs):
    """Plot stellarator boundary at multiple toroidal coordinates.

    Parameters
    ----------
    eq : Equilibrium, Surface
        Object from which to plot.
    phi : float, int or array-like or None
        Values of phi to plot boundary surface at.
        If an integer, plot that many contours linearly spaced in [0,2pi).
        Default is 1 contour for axisymmetric equilibria or 4 for non-axisymmetry.
    plot_axis : bool
        Whether to plot the magnetic axis locations. Default is True.
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
        * ``xlabel_fontsize``: float, fontsize of the x label
        * ``ylabel_fontsize``: float, fontsize of the y label
        * ``legend_kw``: dict, any keyword arguments to be passed to ax.legend()
        * ``cmap``: colormap to use for plotting, discretized into len(phi) colors
        * ``color``: array of colors to use for each phi angle
        * ``ls``: array of line styles to use for each phi angle
        * ``lw``: array of line widths to use for each phi angle
        * ``marker``: str, marker style to use for the axis plotted points
        * ``size``: float, marker size to use for the axis plotted points

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
    .. image:: ../../_static/images/plotting/plot_boundary.png

    .. code-block:: python

        from desc.plotting import plot_boundary
        fig, ax = plot_boundary(eq)

    """
    phi = parse_argname_change(phi, kwargs, "zeta", "phi")

    figsize = kwargs.pop("figsize", None)
    cmap = kwargs.pop("cmap", "hsv")
    colors = kwargs.pop("color", None)
    ls = kwargs.pop("ls", None)
    lw = kwargs.pop("lw", None)
    marker = kwargs.pop("marker", "x")
    size = kwargs.pop("size", 36)
    xlabel_fontsize = kwargs.pop("xlabel_fontsize", None)
    ylabel_fontsize = kwargs.pop("ylabel_fontsize", None)
    legend_kw = kwargs.pop("legend_kw", {})

    assert (
        len(kwargs) == 0
    ), f"plot boundary got unexpected keyword argument: {kwargs.keys()}"

    phi = (1 if eq.N == 0 else 4) if phi is None else phi
    if isinstance(phi, numbers.Integral):
        phi = np.linspace(0, 2 * np.pi / eq.NFP, phi, endpoint=False)
    phi = np.atleast_1d(phi)
    nphi = len(phi)
    # don't plot axis for FourierRZToroidalSurface, since it's not defined.
    plot_axis = plot_axis and eq.L > 0
    rho = np.array([0.0, 1.0]) if plot_axis else np.array([1.0])

    grid_kwargs = {"NFP": 1, "rho": rho, "theta": 100, "zeta": phi}
    grid = _get_grid(**grid_kwargs)
    nr, nt, nz = grid.num_rho, grid.num_theta, grid.num_zeta
    grid = Grid(
        map_coordinates(
            eq,
            grid.nodes,
            ["rho", "theta", "phi"],
            ["rho", "theta", "zeta"],
            period=(np.inf, 2 * np.pi, 2 * np.pi),
            guess=grid.nodes,
        ),
        sort=False,
    )

    if colors is None:
        colors = _get_cmap(cmap)((phi * eq.NFP / (2 * np.pi)) % 1)
    if lw is None:
        lw = 1
    if isinstance(lw, int):
        lw = [lw for _ in range(nz)]
    if ls is None:
        ls = "-"
    if isinstance(ls, str):
        ls = [ls for _ in range(nz)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coords = eq.compute(["R", "Z"], grid=grid)
    R = coords["R"].reshape((nt, nr, nz), order="F")
    Z = coords["Z"].reshape((nt, nr, nz), order="F")

    fig, ax = _format_ax(ax, figsize=figsize, equal=True)

    for i in range(nphi):
        ax.plot(
            R[:, -1, i],
            Z[:, -1, i],
            color=colors[i],
            linestyle=ls[i],
            lw=lw[i],
            label="$\\phi \\cdot N_{{FP}}/2\\pi = {:.2f}$".format(
                eq.NFP * phi[i] / (2 * np.pi)
            ),
        )
        if rho[0] == 0:
            ax.scatter(R[0, 0, i], Z[0, 0, i], color=colors[i], marker=marker, s=size)

    ax.set_xlabel(_AXIS_LABELS_RPZ[0], fontsize=xlabel_fontsize)
    ax.set_ylabel(_AXIS_LABELS_RPZ[2], fontsize=ylabel_fontsize)
    ax.tick_params(labelbottom=True, labelleft=True)

    ax.legend(**legend_kw)
    _set_tight_layout(fig)

    plot_data = {}
    plot_data["R"] = R
    plot_data["Z"] = Z

    if return_data:
        return fig, ax, plot_data

    return fig, ax


def plot_boundaries(
    eqs, labels=None, phi=None, plot_axis=True, ax=None, return_data=False, **kwargs
):
    """Plot stellarator boundaries at multiple toroidal coordinates.

    NOTE: If attempting to plot objects with differing NFP, `phi` must
    be given explicitly.

    Parameters
    ----------
    eqs : array-like of Equilibrium, Surface or EquilibriaFamily
        Equilibria to plot.
    labels : array-like
        Array the same length as eqs of labels to apply to each equilibrium.
    phi : float, int or array-like or None
        Values of phi to plot boundary surface at.
        If an integer, plot that many contours linearly spaced in [0,2pi).
        Default is 1 contour for axisymmetric equilibria or 4 for non-axisymmetry.
    plot_axis : bool
        Whether to plot the magnetic axis locations. Default is True.
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
        * ``xlabel_fontsize``: float, fontsize of the x label
        * ``ylabel_fontsize``: float, fontsize of the y label
        * ``legend``: bool, whether to display legend or not
        * ``legend_kw``: dict, any keyword arguments to be passed to ax.legend()
        * ``cmap``: colormap to use for plotting, discretized into len(eqs) colors
        * ``color``: list of colors to use for each Equilibrium
        * ``ls``: list of str, line styles to use for each Equilibrium
        * ``lw``: list of floats, line widths to use for each Equilibrium
        * ``marker``: str, marker style to use for the axis plotted points
        * ``size``: float, marker size to use for the axis plotted points

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
    .. image:: ../../_static/images/plotting/plot_boundaries.png

    .. code-block:: python

        from desc.plotting import plot_boundaries
        fig, ax = plot_boundaries((eq1, eq2, eq3))

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

    figsize = kwargs.pop("figsize", None)
    cmap = kwargs.pop("cmap", "rainbow")
    colors = kwargs.pop("color", None)
    ls = kwargs.pop("ls", None)
    lw = kwargs.pop("lw", None)
    xlabel_fontsize = kwargs.pop("xlabel_fontsize", None)
    ylabel_fontsize = kwargs.pop("ylabel_fontsize", None)
    marker = kwargs.pop("marker", "x")
    size = kwargs.pop("size", 36)

    phi = (1 if eqs[-1].N == 0 else 4) if phi is None else phi
    if isinstance(phi, numbers.Integral):
        phi = np.linspace(0, 2 * np.pi / eqs[-1].NFP, phi, endpoint=False)
    phi = np.atleast_1d(phi)

    neq = len(eqs)

    if labels is None:
        labels = [str(i) for i in range(neq)]
    if colors is None:
        colors = _get_cmap(cmap, neq)(np.linspace(0, 1, neq))
    if lw is None:
        lw = 1
    if np.isscalar(lw):
        lw = [lw for i in range(neq)]
    if ls is None:
        ls = "-"
    if isinstance(ls, str):
        ls = [ls for i in range(neq)]

    fig, ax = _format_ax(ax, figsize=figsize, equal=True)
    plot_data = {}
    plot_data["R"] = []
    plot_data["Z"] = []

    for i in range(neq):
        # don't plot axis for FourierRZToroidalSurface, since it's not defined.
        plot_axis_i = plot_axis and eqs[i].L > 0
        rho = np.array([0.0, 1.0]) if plot_axis_i else np.array([1.0])

        grid_kwargs = {"NFP": 1, "theta": 100, "zeta": phi, "rho": rho}
        grid = _get_grid(**grid_kwargs)
        nr, nt, nz = grid.num_rho, grid.num_theta, grid.num_zeta
        grid = Grid(
            map_coordinates(
                eqs[i],
                grid.nodes,
                ["rho", "theta", "phi"],
                ["rho", "theta", "zeta"],
                period=(np.inf, 2 * np.pi, 2 * np.pi),
                guess=grid.nodes,
            ),
            sort=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            coords = eqs[i].compute(["R", "Z"], grid=grid)
        R = coords["R"].reshape((nt, nr, nz), order="F")
        Z = coords["Z"].reshape((nt, nr, nz), order="F")

        plot_data["R"].append(R)
        plot_data["Z"].append(Z)

        for j in range(nz):
            (line,) = ax.plot(
                R[:, -1, j], Z[:, -1, j], color=colors[i], linestyle=ls[i], lw=lw[i]
            )
            if rho[0] == 0:
                ax.scatter(
                    R[0, 0, j], Z[0, 0, j], color=colors[i], marker=marker, s=size
                )

            if j == 0:
                line.set_label(labels[i])

    ax.set_xlabel(_AXIS_LABELS_RPZ[0], fontsize=xlabel_fontsize)
    ax.set_ylabel(_AXIS_LABELS_RPZ[2], fontsize=ylabel_fontsize)
    ax.tick_params(labelbottom=True, labelleft=True)

    if any(labels) and kwargs.pop("legend", True):
        ax.legend(**kwargs.pop("legend_kw", {}))
    _set_tight_layout(fig)

    assert (
        len(kwargs) == 0
    ), f"plot boundaries got unexpected keyword argument: {kwargs.keys()}"

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

    rows = np.floor(np.sqrt(nphi)).astype(int)
    cols = np.ceil(nphi / rows).astype(int)

    figw = 4 * cols
    figh = 5 * rows
    if figsize is None:
        figsize = (figw, figh)
    fig, ax = _format_ax(ax, rows=rows, cols=cols, figsize=figsize, equal=True)
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
            title_fontsize=title_fontsize,
            xlabel_fontsize=xlabel_fontsize,
            ylabel_fontsize=ylabel_fontsize,
            return_data=True,
        )
        for key in _plot_data.keys():
            plot_data[key].append(_plot_data[key])

    if any(labels) and kwargs.pop("legend", True):
        fig.legend(**kwargs.pop("legend_kw", {}))

    assert (
        len(kwargs) == 0
    ), f"plot_comparison got unexpected keyword argument: {kwargs.keys()}"

    if return_data:
        return fig, ax, plot_data

    return fig, ax


def plot_coils(coils, grid=None, fig=None, return_data=False, **kwargs):
    """Create 3D plot of coil geometry.

    Parameters
    ----------
    coils : Coil, CoilSet, Curve, or iterable
        Coil or coils to plot.
    grid : Grid, optional
        Grid to use for evaluating geometry
    fig : plotly.graph_objs._figure.Figure, optional
        Figure to plot on.
    return_data : bool
        If True, return the data plotted as well as fig,ax
    **kwargs : dict, optional
        Specify properties of the figure, axis, and plot appearance e.g.::

            plot_X(figsize=(4,6), color="darkgrey")

        Valid keyword arguments are:

        * ``unique``: bool, only plots unique coils from a CoilSet if True
        * ``figsize``: tuple of length 2, the size of the figure in inches
        * ``lw``: float, linewidth of plotted coils
        * ``ls``: str, linestyle of plotted coils
        * ``color``: str, color of plotted coils
        * ``showgrid``: Bool, whether or not to show the coordinate grid lines.
          True by default.
        * ``showticklabels``: Bool, whether or not to show the coordinate tick labels.
          True by default.
        * ``showaxislabels``: Bool, whether or not to show the coordinate axis labels.
          True by default.
        * ``zeroline``: Bool, whether or not to show the zero coordinate axis lines.
          True by default.

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        Figure being plotted to
    plot_data : dict
        Dictionary of the data plotted, only returned if ``return_data=True``

    """
    lw = kwargs.pop("lw", 5)
    ls = kwargs.pop("ls", "solid")
    figsize = kwargs.pop("figsize", (10, 10))
    color = kwargs.pop("color", "black")
    unique = kwargs.pop("unique", False)
    showgrid = kwargs.pop("showgrid", True)
    zeroline = kwargs.pop("zeroline", True)
    showticklabels = kwargs.pop("showticklabels", True)
    showaxislabels = kwargs.pop("showaxislabels", True)
    errorif(
        len(kwargs) != 0,
        ValueError,
        f"plot_coils got unexpected keyword argument: {kwargs.keys()}",
    )
    errorif(
        not isinstance(coils, _Coil),
        ValueError,
        "Expected `coils` to be of type `_Coil`, instead got type" f" {type(coils)}",
    )

    if not isinstance(lw, (list, tuple)):
        lw = [lw]
    if not isinstance(ls, (list, tuple)):
        ls = [ls]
    if not isinstance(color, (list, tuple)):
        color = [color]
    if grid is None:
        grid = LinearGrid(N=400, endpoint=True)

    def flatten_coils(coilset):
        if hasattr(coilset, "__len__"):
            if hasattr(coilset, "_NFP") and hasattr(coilset, "_sym"):
                if not unique and (coilset.NFP > 1 or coilset.sym):
                    # plot all coils for symmetric coil sets
                    coilset = CoilSet.from_symmetry(
                        coilset, NFP=coilset.NFP, sym=coilset.sym
                    )
            return [a for i in coilset for a in flatten_coils(i)]
        else:
            return [coilset]

    coils_list = flatten_coils(coils)
    plot_data = {}
    plot_data["X"] = []
    plot_data["Y"] = []
    plot_data["Z"] = []

    if fig is None:
        fig = go.Figure()

    for i, coil in enumerate(coils_list):
        x, y, z = coil.compute("x", grid=grid, basis="xyz")["x"].T
        current = getattr(coil, "current", np.nan)
        plot_data["X"].append(x)
        plot_data["Y"].append(y)
        plot_data["Z"].append(z)

        trace = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            marker=dict(
                size=0,
                opacity=0,
            ),
            line=dict(
                color=color[i % len(color)],
                width=lw[i % len(lw)],
                dash=ls[i % len(ls)],
            ),
            showlegend=False,
            name=coil.name or f"CoilSet[{i}]",
            hovertext=f"Current = {current} (A)",
        )

        fig.add_trace(trace)
    xaxis_title = "X (m)" if showaxislabels else ""
    yaxis_title = "Y (m)" if showaxislabels else ""
    zaxis_title = "Z (m)" if showaxislabels else ""
    fig.update_layout(
        scene=dict(
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            zaxis_title=zaxis_title,
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
            aspectmode="data",
        ),
        width=figsize[0] * dpi,
        height=figsize[1] * dpi,
    )
    if return_data:
        return fig, plot_data
    return fig


def plot_boozer_modes(  # noqa: C901
    eq,
    log=True,
    B0=True,
    norm=False,
    num_modes=10,
    rho=None,
    helicity=None,
    max_only=False,
    ax=None,
    return_data=False,
    **kwargs,
):
    """Plot Fourier harmonics of :math:`|B|` in Boozer coordinates.

    Parameters
    ----------
    eq : Equilibrium
        Object from which to plot.
    log : bool, optional
        Whether to use a log scale.
    B0 : bool, optional
        Whether to include the m=n=0 mode.
    norm : bool, optional
        Whether to normalize the magnitudes such that B0=1 Tesla.
    num_modes : int, optional
        How many modes to include. Use -1 for all modes.
    rho : int or ndarray, optional
        Radial coordinates of the flux surfaces to evaluate at,
        or number of surfaces in (0,1]
    helicity : None or tuple of int
        If a tuple, the (M,N) helicity of the field, only symmetry breaking modes are
        plotted. If None, plot all modes.
    max_only : bool
        If True, only plot the maximum of the symmetry breaking modes. Helicity must
        be specified.
    ax : matplotlib AxesSubplot, optional
        Axis to plot on.
    return_data : bool
        If True, return the data plotted as well as fig,ax
    **kwargs : dict, optional
        Specify properties of the figure, axis, and plot appearance e.g.::

            plot_X(figsize=(4,6))

        Valid keyword arguments are:

        * ``figsize``: tuple of length 2, the size of the figure (to be passed to
          matplotlib)
        * ``lw``: float, linewidth
        * ``ls``: str, linestyle
        * ``legend``: bool, whether to display legend or not
        * ``legend_kw``: dict, any keyword arguments to be passed to ax.legend()
        * ``xlabel_fontsize``: float, fontsize of the xlabel
        * ``ylabel_fontsize``: float, fontsize of the ylabel
        * ``label`` : str, label to apply. Only used if ``max_only`` is True.
        * ``color`` : str, color for plotted line. Only used if ``max_only`` is True.
        * ``M_booz`` : int, poloidal resolution to use for Boozer transform.
        * ``N_booz`` : int, toroidal resolution to use for Boozer transform.

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
    .. image:: ../../_static/images/plotting/plot_boozer_modes.png

    .. code-block:: python

        from desc.plotting import plot_boozer_modes
        fig, ax = plot_boozer_modes(eq)

    """
    if rho is None:
        rho = np.linspace(1, 0, num=20, endpoint=False)
    elif np.isscalar(rho) and rho > 1:
        rho = np.linspace(1, 0, num=rho, endpoint=False)

    rho = np.sort(rho)
    M_booz = kwargs.pop("M_booz", 2 * eq.M)
    N_booz = kwargs.pop("N_booz", 2 * eq.N)
    linestyle = kwargs.pop("ls", "-")
    linewidth = kwargs.pop("lw", 2)
    xlabel_fontsize = kwargs.pop("xlabel_fontsize", None)
    ylabel_fontsize = kwargs.pop("ylabel_fontsize", None)

    basis = get_transforms(
        "|B|_mn_B", obj=eq, grid=Grid(np.array([])), M_booz=M_booz, N_booz=N_booz
    )["B"].basis
    if helicity:
        matrix, modes, symidx = ptolemy_linear_transform(
            basis.modes, helicity=helicity, NFP=eq.NFP
        )
    else:
        matrix, modes = ptolemy_linear_transform(basis.modes)

    grid = LinearGrid(M=2 * eq.M_grid, N=2 * eq.N_grid, NFP=eq.NFP, rho=rho)
    transforms = get_transforms(
        "|B|_mn_B", obj=eq, grid=grid, M_booz=M_booz, N_booz=N_booz
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = eq.compute("|B|_mn_B", grid=grid, transforms=transforms)
    B_mn = data["|B|_mn_B"].reshape((len(rho), -1))
    B_mn = np.atleast_2d(matrix @ B_mn.T).T

    zidx = np.where((modes[:, 1:] == np.array([[0, 0]])).all(axis=1))[0]
    if norm:
        B_mn = B_mn / B_mn[:, zidx]
    if helicity:
        B_mn = B_mn[:, symidx]
        modes = modes[symidx, :]
    elif not B0:
        B_mn = np.delete(B_mn, zidx, axis=-1)
        modes = np.delete(modes, zidx, axis=0)

    if max_only:
        assert helicity is not None
        B_mn = np.max(np.abs(B_mn), axis=1)
        modes = None
    else:
        idx = np.argsort(np.mean(np.abs(B_mn), axis=0))
        idx = idx[-1::-1] if (num_modes == -1) else idx[-1 : -num_modes - 1 : -1]
        B_mn = B_mn[:, idx]
        modes = modes[idx, :]

    fig, ax = _format_ax(ax, figsize=kwargs.pop("figsize", None))

    plot_op = ax.semilogy if log else ax.plot
    B_mn = np.abs(B_mn) if log else B_mn

    if max_only:
        plot_op(
            rho,
            B_mn,
            label=kwargs.pop("label", ""),
            color=kwargs.pop("color", "k"),
            linestyle=linestyle,
            linewidth=linewidth,
        )
    else:
        for i, (L, M, N) in enumerate(modes):
            N *= int(eq.NFP)
            plot_op(
                rho,
                B_mn[:, i],
                label="M={}, N={}{}".format(
                    M, N, "" if eq.sym else (" (cos)" if L > 0 else " (sin)")
                ),
                linestyle=linestyle,
                linewidth=linewidth,
            )

    plot_data = {
        "|B|_mn_B": B_mn,
        "B modes": modes,
        "rho": rho,
    }

    ax.set_xlabel(_AXIS_LABELS_RTZ[0], fontsize=xlabel_fontsize)
    if max_only:
        ylabel = r"Max symmetry breaking Boozer $B_{M,N}$"
    elif helicity:
        ylabel = r"Symmetry breaking Boozer $B_{M,N}$"
    else:
        ylabel = r"$B_{M,N}$ in Boozer coordinates"
    ylabel += r" (normalized)" if norm else r" $(T)$"
    ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)

    if kwargs.pop("legend", True):
        ax.legend(**kwargs.pop("legend_kw", {"loc": "lower right"}))

    assert (
        len(kwargs) == 0
    ), f"plot boozer modes got unexpected keyword argument: {kwargs.keys()}"

    _set_tight_layout(fig)
    if return_data:
        return fig, ax, plot_data

    return fig, ax


def plot_boozer_surface(
    thing,
    grid_compute=None,
    grid_plot=None,
    rho=1,
    fill=False,
    ncontours=30,
    fieldlines=0,
    ax=None,
    return_data=False,
    **kwargs,
):
    """Plot :math:`|B|` on a surface vs the Boozer poloidal and toroidal angles.

    Parameters
    ----------
    thing : Equilibrium or OmnigenousField
        Object from which to plot.
    grid_compute : Grid, optional
        Grid to use for computing boozer spectrum
    grid_plot : Grid, optional
        Grid to plot on.
    rho : float, optional
        Radial coordinate of flux surface. Used only if grids are not specified.
    fill : bool, optional
        Whether the contours are filled, i.e. whether to use `contourf` or `contour`.
    ncontours : int, optional
        Number of contours to plot.
    fieldlines : int, optional
        Number of (linearly spaced) magnetic fieldlines to plot. Default is 0 (none).
    ax : matplotlib AxesSubplot, optional
        Axis to plot on.
    return_data : bool
        If True, return the data plotted as well as fig,ax
    **kwargs : dict, optional
        Specify properties of the figure, axis, and plot appearance e.g.::

            plot_X(figsize=(4,6),cmap="plasma")

        Valid keyword arguments are:

        * ``iota``: rotational transform, used when `thing` is an OmnigenousField
        * ``figsize``: tuple of length 2, the size of the figure (to be passed to
          matplotlib)
        * ``cmap``: str, matplotlib colormap scheme to use, passed to ax.contourf
        * ``levels``: int or array-like, passed to contourf
        * ``title_fontsize``: integer, font size of the title
        * ``xlabel_fontsize``: float, fontsize of the xlabel
        * ``ylabel_fontsize``: float, fontsize of the ylabel

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure being plotted to
    ax : matplotlib.axes.Axes or ndarray of Axes
        Axes being plotted to
    plot_data : dict
        Dictionary of the data plotted, only returned if ``return_data=True``

    Examples
    --------
    .. image:: ../../_static/images/plotting/plot_boozer_surface.png

    .. code-block:: python

        from desc.plotting import plot_boozer_surface
        fig, ax = plot_boozer_surface(eq)

    .. image:: ../../_static/images/plotting/plot_omnigenous_field.png

    .. code-block:: python

        from desc.plotting import plot_boozer_surface
        fig, ax = plot_boozer_surface(field, iota=0.32)

    """
    eq_switch = True
    if hasattr(thing, "_x_lmn"):
        eq_switch = False  # thing is an OmnigenousField, not an Equilibrium

    # default grids
    if grid_compute is None:
        # grid_compute only used for Equilibrium, not OmnigenousField
        grid_kwargs = {
            "rho": rho,
            "M": 4 * getattr(thing, "M", 1),
            "N": 4 * getattr(thing, "N", 1),
            "NFP": thing.NFP,
            "endpoint": False,
        }
        grid_compute = _get_grid(**grid_kwargs)
    if grid_plot is None:
        grid_kwargs = {
            "rho": rho,
            "theta": 91,
            "zeta": 91,
            "NFP": thing.NFP,
            "endpoint": eq_switch,
        }
        grid_plot = _get_grid(**grid_kwargs)

    # compute
    if eq_switch:  # Equilibrium
        M_booz = kwargs.pop("M_booz", 2 * thing.M)
        N_booz = kwargs.pop("N_booz", 2 * thing.N)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = thing.compute(
                "|B|_mn_B", grid=grid_compute, M_booz=M_booz, N_booz=N_booz
            )
        B_transform = get_transforms(
            "|B|_mn_B", obj=thing, grid=grid_plot, M_booz=M_booz, N_booz=N_booz
        )["B"]
        B = B_transform.transform(data["|B|_mn_B"]).reshape(
            (grid_plot.num_theta, grid_plot.num_zeta), order="F"
        )
        theta_B = (
            grid_plot.nodes[:, 1]
            .reshape((grid_plot.num_theta, grid_plot.num_zeta), order="F")
            .squeeze()
        )
        zeta_B = (
            grid_plot.nodes[:, 2]
            .reshape((grid_plot.num_theta, grid_plot.num_zeta), order="F")
            .squeeze()
        )
        iota = grid_compute.compress(data["iota"])
    else:  # OmnigenousField
        iota = kwargs.pop("iota", None)
        errorif(iota is None, ValueError, "iota must be supplied for OmnigenousField")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = thing.compute(
                ["theta_B", "zeta_B", "|B|"],
                grid=grid_plot,
                helicity=thing.helicity,
                iota=iota,
            )
        B = data["|B|"]
        theta_B = np.mod(data["theta_B"], 2 * np.pi)
        zeta_B = np.mod(data["zeta_B"], 2 * np.pi / thing.NFP)

    fig, ax = _format_ax(ax, figsize=kwargs.pop("figsize", None))
    divider = make_axes_locatable(ax)

    contourf_kwargs = {
        "norm": matplotlib.colors.Normalize(),
        "levels": kwargs.pop(
            "levels", np.linspace(np.nanmin(B), np.nanmax(B), ncontours)
        ),
        "cmap": kwargs.pop("cmap", "jet"),
        "extend": "both",
    }

    title_fontsize = kwargs.pop("title_fontsize", None)
    xlabel_fontsize = kwargs.pop("xlabel_fontsize", None)
    ylabel_fontsize = kwargs.pop("ylabel_fontsize", None)

    assert (
        len(kwargs) == 0
    ), f"plot_boozer_surface got unexpected keyword argument: {kwargs.keys()}"

    # plot
    op = ("" if eq_switch else "tri") + "contour" + ("f" if fill else "")
    im = getattr(ax, op)(zeta_B, theta_B, B, **contourf_kwargs)

    cax_kwargs = {"size": "5%", "pad": 0.05}
    cax = divider.append_axes("right", **cax_kwargs)
    cbar = fig.colorbar(im, cax=cax)
    cbar.update_ticks()

    if fieldlines:
        theta0 = np.linspace(0, 2 * np.pi, fieldlines, endpoint=False)
        zeta = np.linspace(0, 2 * np.pi / grid_plot.NFP, 100)
        alpha = np.atleast_2d(theta0) + iota * np.atleast_2d(zeta).T
        alpha1 = np.where(np.logical_and(alpha >= 0, alpha <= 2 * np.pi), alpha, np.nan)
        alpha2 = np.where(
            np.logical_or(alpha < 0, alpha > 2 * np.pi),
            alpha % (sign(iota) * 2 * np.pi) + (sign(iota) < 0) * (2 * np.pi),
            np.nan,
        )
        alphas = np.hstack((alpha1, alpha2))
        ax.plot(zeta, alphas, color="k", ls="-", lw=2)

    ax.set_xlim([0, 2 * np.pi / thing.NFP])
    ax.set_ylim([0, 2 * np.pi])

    ax.set_xlabel(r"$\zeta_{Boozer}$", fontsize=xlabel_fontsize)
    ax.set_ylabel(r"$\theta_{Boozer}$", fontsize=ylabel_fontsize)
    ax.set_title(r"$|\mathbf{B}|~(T)$", fontsize=title_fontsize)

    _set_tight_layout(fig)

    if return_data:
        plot_data = {"theta_B": theta_B, "zeta_B": zeta_B, "|B|": B}
        return fig, ax, plot_data

    return fig, ax


def plot_qs_error(  # noqa: 16 fxn too complex
    eq,
    log=True,
    fB=True,
    fC=True,
    fT=True,
    helicity=(1, 0),
    rho=None,
    ax=None,
    return_data=False,
    **kwargs,
):
    """Plot quasi-symmetry errors f_B, f_C, and f_T as normalized flux functions.

    Parameters
    ----------
    eq : Equilibrium
        Object from which to plot.
    log : bool, optional
        Whether to use a log scale.
    fB : bool, optional
        Whether to include the Boozer coordinates QS error.
    fC : bool, optional
        Whether to include the flux function QS error.
    fT : bool, optional
        Whether to include the triple product QS error.
    helicity : tuple, int
        Type of quasi-symmetry (M, N).
    rho : int or ndarray, optional
        Radial coordinates of the flux surfaces to evaluate at,
        or number of surfaces in (0,1]
    ax : matplotlib AxesSubplot, optional
        Axis to plot on.
    return_data : bool
        If True, return the data plotted as well as fig,ax
    **kwargs : dict, optional
        Specify properties of the figure, axis, and plot appearance e.g.::

            plot_X(figsize=(4,6))

        Valid keyword arguments are:

        * ``figsize``: tuple of length 2, the size of the figure (to be passed to
          matplotlib)
        * ``ls``: list of strs of length 3, linestyles to use for the 3 different
          qs metrics
        * ``lw``: list of float of length 3, linewidths to use for the 3 different
          qs metrics
        * ``color``: list of strs of length 3, colors to use for the 3 different
          qs metrics
        * ``marker``: list of strs of length 3, markers to use for the 3 different
          qs metrics
        * ``labels``:  list of strs of length 3, labels to use for the 3 different
          qs metrics
        * ``ylabel``: str, ylabel to use for plot
        * ``legend``: bool, whether to display legend or not
        * ``legend_kw``: dict, any keyword arguments to be passed to ax.legend()
        * ``xlabel_fontsize``: float, fontsize of the xlabel
        * ``ylabel_fontsize``: float, fontsize of the ylabel
        * ``labels``: list of strs of length 3, labels to apply to each QS error metric

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
    .. image:: ../../_static/images/plotting/plot_qs_error.png

    .. code-block:: python

        from desc.plotting import plot_qs_error
        fig, ax = plot_qs_error(eq, helicity=(1, eq.NFP), log=True)

    """
    colors = kwargs.pop("color", ["r", "b", "g"])
    markers = kwargs.pop("marker", ["o", "o", "o"])
    labels = kwargs.pop("labels", [r"$\hat{f}_B$", r"$\hat{f}_C$", r"$\hat{f}_T$"])
    colors = parse_argname_change(colors, kwargs, "colors", "color")
    markers = parse_argname_change(markers, kwargs, "markers", "marker")

    if rho is None:
        rho = np.linspace(1, 0, num=20, endpoint=False)
    elif np.isscalar(rho) and rho > 1:
        rho = np.linspace(1, 0, num=rho, endpoint=False)
    rho = np.sort(rho)

    fig, ax = _format_ax(ax, figsize=kwargs.pop("figsize", None))

    M_booz = kwargs.pop("M_booz", 2 * eq.M)
    N_booz = kwargs.pop("N_booz", 2 * eq.N)
    ls = kwargs.pop("ls", ["-", "-", "-"])
    lw = kwargs.pop("lw", [1, 1, 1])
    ylabel = kwargs.pop("ylabel", False)
    xlabel_fontsize = kwargs.pop("xlabel_fontsize", None)
    ylabel_fontsize = kwargs.pop("ylabel_fontsize", None)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = eq.compute(["R0", "|B|"])
    R0 = data["R0"]
    B0 = np.mean(data["|B|"] * data["sqrt(g)"]) / np.mean(data["sqrt(g)"])

    plot_data = {"rho": rho}

    grid = LinearGrid(M=2 * eq.M_grid, N=2 * eq.N_grid, NFP=eq.NFP, rho=rho)
    names = []
    if fB:
        names += ["|B|_mn_B"]
        transforms = get_transforms(
            "|B|_mn_B", obj=eq, grid=grid, M_booz=M_booz, N_booz=N_booz
        )
        matrix, modes, idx = ptolemy_linear_transform(
            transforms["B"].basis.modes,
            helicity=helicity,
            NFP=transforms["B"].basis.NFP,
        )
    if fC or fT:
        names += ["sqrt(g)"]
    if fC:
        names += ["f_C"]
    if fT:
        names += ["f_T"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = eq.compute(
            names, grid=grid, M_booz=M_booz, N_booz=N_booz, helicity=helicity
        )

    if fB:
        B_mn = data["|B|_mn_B"].reshape((len(rho), -1))
        B_mn = (matrix @ B_mn.T).T
        f_B = np.sqrt(np.sum(B_mn[:, idx] ** 2, axis=-1)) / np.sqrt(
            np.sum(B_mn**2, axis=-1)
        )
        plot_data["f_B"] = f_B
    if fC:
        sqrtg = grid.meshgrid_reshape(data["sqrt(g)"], "rtz")
        f_C = grid.meshgrid_reshape(data["f_C"], "rtz")
        f_C = (
            np.mean(np.abs(f_C) * sqrtg, axis=(1, 2))
            / np.mean(sqrtg, axis=(1, 2))
            / B0**3
        )
        plot_data["f_C"] = f_C
    if fT:
        sqrtg = grid.meshgrid_reshape(data["sqrt(g)"], "rtz")
        f_T = grid.meshgrid_reshape(data["f_T"], "rtz")
        f_T = (
            np.mean(np.abs(f_T) * sqrtg, axis=(1, 2))
            / np.mean(sqrtg, axis=(1, 2))
            * R0**2
            / B0**4
        )
        plot_data["f_T"] = f_T

    plot_op = ax.semilogy if log else ax.plot

    if fB:
        plot_op(
            rho,
            f_B,
            ls=ls[0 % len(ls)],
            c=colors[0 % len(colors)],
            marker=markers[0 % len(markers)],
            label=labels[0 % len(labels)],
            lw=lw[0 % len(lw)],
        )
    if fC:
        plot_op(
            rho,
            f_C,
            ls=ls[1 % len(ls)],
            c=colors[1 % len(colors)],
            marker=markers[1 % len(markers)],
            label=labels[1 % len(labels)],
            lw=lw[1 % len(lw)],
        )
    if fT:
        plot_op(
            rho,
            f_T,
            ls=ls[2 % len(ls)],
            c=colors[2 % len(colors)],
            marker=markers[2 % len(markers)],
            label=labels[2 % len(labels)],
            lw=lw[2 % len(lw)],
        )

    ax.set_xlabel(_AXIS_LABELS_RTZ[0], fontsize=xlabel_fontsize)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)

    if kwargs.pop("legend", True):
        ax.legend(**kwargs.pop("legend_kw", {"loc": "center right"}))

    assert (
        len(kwargs) == 0
    ), f"plot qs error got unexpected keyword argument: {kwargs.keys()}"

    _set_tight_layout(fig)
    if return_data:
        return fig, ax, plot_data

    return fig, ax


def plot_grid(grid, return_data=False, **kwargs):
    """Plot the location of collocation nodes on the zeta=0 plane.

    Parameters
    ----------
    grid : Grid
        Grid to plot.
    return_data : bool
        If True, return the data plotted as well as fig,ax
    **kwargs : dict, optional
        Specify properties of the figure, axis, and plot appearance e.g.::

            plot_X(figsize=(4,6))

        Valid keyword arguments are:

        * ``figsize``: tuple of length 2, the size of the figure (to be passed to
          matplotlib)
        * ``title_fontsize``: integer, font size of the title

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
    .. image:: ../../_static/images/plotting/plot_grid.png

    .. code-block:: python

        from desc.plotting import plot_grid
        from desc.grid import ConcentricGrid
        grid = ConcentricGrid(L=20, M=10, N=1, node_pattern="jacobi")
        fig, ax = plot_grid(grid)

    """
    fig = plt.figure(figsize=kwargs.pop("figsize", (4, 4)))
    ax = plt.subplot(projection="polar")
    title_fontsize = kwargs.pop("title_fontsize", None)

    assert (
        len(kwargs) == 0
    ), f"plot_grid got unexpected keyword argument: {kwargs.keys()}"

    # node locations
    nodes = grid.nodes[grid.nodes[:, 2] == 0]
    ax.scatter(nodes[:, 1], nodes[:, 0], s=4)
    ax.set_ylim(0, 1)
    ax.set_xticks(
        [
            0,
            np.pi / 4,
            np.pi / 2,
            3 / 4 * np.pi,
            np.pi,
            5 / 4 * np.pi,
            3 / 2 * np.pi,
            7 / 4 * np.pi,
        ]
    )
    ax.set_xticklabels(
        [
            "$0$",
            r"$\frac{\pi}{4}$",
            r"$\frac{\pi}{2}$",
            r"$\frac{3\pi}{4}$",
            r"$\pi$",
            r"$\frac{5\pi}{4}$",
            r"$\frac{3\pi}{2}$",
            r"$\frac{7\pi}{4}$",
        ]
    )
    ax.set_yticklabels([])
    if grid.__class__.__name__ in ["LinearGrid", "Grid", "QuadratureGrid"]:
        ax.set_title(
            "{}, $L={}$, $M={}, pattern: {}$".format(
                grid.__class__.__name__, grid.L, grid.M, grid.node_pattern
            ),
            pad=20,
        )
    if grid.__class__.__name__ in ["ConcentricGrid"]:
        ax.set_title(
            "{}, $M={}$, pattern: {}".format(
                grid.__class__.__name__,
                grid.M,
                grid.node_pattern,
            ),
            pad=20,
            fontsize=title_fontsize,
        )
    _set_tight_layout(fig)

    plot_data = {"rho": nodes[:, 0], "theta": nodes[:, 1]}

    if return_data:
        return fig, ax, plot_data

    return fig, ax


def plot_basis(basis, return_data=False, **kwargs):
    """Plot basis functions.

    Parameters
    ----------
    basis : Basis
        basis to plot
    return_data : bool
        If True, return the data plotted as well as fig,ax
    **kwargs : dict, optional
        Specify properties of the figure, axis, and plot appearance e.g.::

            plot_X(figsize=(4,6),cmap="plasma")

        Valid keyword arguments are:

        * ``figsize``: tuple of length 2, the size of the figure (to be passed to
          matplotlib)
        * ``cmap``: str, matplotlib colormap scheme to use, passed to ax.contourf
        * ``title_fontsize``: integer, font size of the title

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure being plotted to.
    ax : matplotlib.axes.Axes, ndarray of axes, or dict of axes
        Axes being plotted to. A single axis is used for 1d basis functions,
        2d or 3d bases return an ndarray or dict of axes.    return_data : bool
        if True, return the data plotted as well as fig,ax
    plot_data : dict
        Dictionary of the data plotted, only returned if ``return_data=True``

    Examples
    --------
    .. image:: ../../_static/images/plotting/plot_basis.png

    .. code-block:: python

        from desc.plotting import plot_basis
        from desc.basis import DoubleFourierSeries
        basis = DoubleFourierSeries(M=3, N=2)
        fig, ax = plot_basis(basis)

    """
    title_fontsize = kwargs.pop("title_fontsize", None)

    # TODO(#1377): add all other Basis classes
    if basis.__class__.__name__ == "PowerSeries":
        grid = LinearGrid(rho=100, endpoint=True)
        r = grid.nodes[:, 0]
        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (6, 4)))

        f = basis.evaluate(grid.nodes)
        plot_data = {"l": basis.modes[:, 0], "amplitude": [], "rho": r}

        for fi, l in zip(f.T, basis.modes[:, 0]):
            ax.plot(r, fi, label="$l={:d}$".format(int(l)))
            plot_data["amplitude"].append(fi)
        ax.set_xlabel("$\\rho$")
        ax.set_ylabel("$f_l(\\rho)$")
        ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_title(
            "{}, $L={}$".format(basis.__class__.__name__, basis.L),
            fontsize=title_fontsize,
        )
        _set_tight_layout(fig)
        if return_data:
            return fig, ax, plot_data

        return fig, ax

    elif basis.__class__.__name__ == "FourierSeries":
        grid = LinearGrid(zeta=100, NFP=basis.NFP, endpoint=True)
        z = grid.nodes[:, 2]
        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (6, 4)))

        f = basis.evaluate(grid.nodes)
        plot_data = {"n": basis.modes[:, 2], "amplitude": [], "zeta": z}

        for fi, n in zip(f.T, basis.modes[:, 2]):
            ax.plot(z, fi, label="$n={:d}$".format(int(n)))
            plot_data["amplitude"].append(fi)

        ax.set_xlabel("$\\zeta$")
        ax.set_ylabel("$f_n(\\zeta)$")
        ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
        ax.set_xticks([0, np.pi / basis.NFP, 2 * np.pi / basis.NFP])
        ax.set_xticklabels(["$0$", "$\\pi/N_{{FP}}$", "$2\\pi/N_{{FP}}$"])
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax.set_title(
            "{}, $N={}$, $N_{{FP}}={}$".format(
                basis.__class__.__name__, basis.N, basis.NFP
            ),
            fontsize=title_fontsize,
        )
        _set_tight_layout(fig)
        if return_data:
            return fig, ax, plot_data

        return fig, ax

    elif basis.__class__.__name__ == "DoubleFourierSeries":
        nmax = abs(basis.modes[:, 2]).max()
        mmax = abs(basis.modes[:, 1]).max()
        grid = LinearGrid(theta=100, zeta=100, NFP=basis.NFP, endpoint=True)
        t = grid.nodes[:, 1].reshape((grid.num_theta, grid.num_zeta))
        z = grid.nodes[:, 2].reshape((grid.num_theta, grid.num_zeta))
        fig = plt.figure(
            figsize=kwargs.get("figsize", (nmax * 4 + 1, mmax * 4 + 1)),
        )
        wratios = np.ones(2 * nmax + 2)
        wratios[-1] = kwargs.get("cbar_ratio", 0.25)
        hratios = np.ones(2 * mmax + 2)
        hratios[0] = kwargs.get("title_ratio", 0.1)
        gs = matplotlib.gridspec.GridSpec(
            2 * mmax + 2, 2 * nmax + 2, width_ratios=wratios, height_ratios=hratios
        )
        ax = np.empty((2 * mmax + 1, 2 * nmax + 1), dtype=object)
        f = basis.evaluate(grid.nodes)
        plot_data = {
            "m": basis.modes[:, 1],
            "n": basis.modes[:, 2],
            "amplitude": [],
            "zeta": z,
            "theta": t,
        }

        for fi, m, n in zip(f.T, basis.modes[:, 1], basis.modes[:, 2]):
            ax[mmax + m, nmax + n] = plt.subplot(gs[mmax + m + 1, n + nmax])
            ax[mmax + m, nmax + n].set_xticks(
                [
                    0,
                    np.pi / basis.NFP / 2,
                    np.pi / basis.NFP,
                    3 / 2 * np.pi / basis.NFP,
                    2 * np.pi / basis.NFP,
                ]
            )
            ax[mmax + m, 0].set_yticks([0, np.pi / 2, np.pi, 3 / 2 * np.pi, 2 * np.pi])
            ax[mmax + m, nmax + n].set_xticklabels([])
            ax[mmax + m, nmax + n].set_yticklabels([])
            im = ax[mmax + m, nmax + n].contourf(
                z,
                t,
                fi.reshape((grid.num_theta, grid.num_zeta)),
                levels=100,
                vmin=-1,
                vmax=1,
                cmap=kwargs.get("cmap", "coolwarm"),
            )
            plot_data["amplitude"].append(fi.reshape((grid.num_theta, grid.num_zeta)))

            if m == mmax:
                ax[mmax + m, nmax + n].set_xlabel(
                    "$\\zeta$ \n $n={}$".format(n), fontsize=10
                )
                ax[mmax + m, nmax + n].set_xticklabels(
                    ["$0$", None, "$\\pi/N_{{FP}}$", None, "$2\\pi/N_{{FP}}$"],
                    fontsize=8,
                )
            if n + nmax == 0:
                ax[mmax + m, 0].set_ylabel("$m={}$ \n $\\theta$".format(m), fontsize=10)
                ax[mmax + m, 0].set_yticklabels(
                    ["$0$", None, "$\\pi$", None, "$2\\pi$"], fontsize=8
                )
        cb_ax = plt.subplot(gs[:, -1])
        cbar = fig.colorbar(im, cax=cb_ax)
        cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
        fig.suptitle(
            "{}, $M={}$, $N={}$, $N_{{FP}}={}$".format(
                basis.__class__.__name__, basis.M, basis.N, basis.NFP
            ),
            y=0.98,
            fontsize=title_fontsize,
        )
        if return_data:
            return fig, ax, plot_data

        return fig, ax
    elif basis.__class__.__name__ in ["ZernikePolynomial", "FourierZernikeBasis"]:
        lmax = abs(basis.modes[:, 0]).max().astype(int)
        mmax = abs(basis.modes[:, 1]).max().astype(int)

        grid = LinearGrid(rho=100, theta=100, endpoint=True)
        r = grid.nodes[grid.unique_rho_idx, 0]
        v = grid.nodes[grid.unique_theta_idx, 1]

        fig = plt.figure(figsize=kwargs.get("figsize", (3 * mmax, 3 * lmax / 2)))

        plot_data = {"amplitude": [], "rho": r, "theta": v}

        ax = {i: {} for i in range(lmax + 1)}
        ratios = np.ones(2 * (mmax + 1) + 1)
        ratios[-1] = kwargs.get("cbar_ratio", 0.25)
        gs = matplotlib.gridspec.GridSpec(
            lmax + 2, 2 * (mmax + 1) + 1, width_ratios=ratios
        )

        modes = basis.modes[basis.modes[:, 2] == 0]
        plot_data["l"] = basis.modes[:, 0]
        plot_data["m"] = basis.modes[:, 1]
        Zs = basis.evaluate(grid.nodes, modes=modes)
        for i, (l, m) in enumerate(
            zip(modes[:, 0].astype(int), modes[:, 1].astype(int))
        ):
            Z = Zs[:, i].reshape((grid.num_rho, grid.num_theta))
            ax[l][m] = plt.subplot(
                gs[l + 1, m + mmax : m + mmax + 2], projection="polar"
            )
            ax[l][m].set_title("$l={}, m={}$".format(l, m))
            ax[l][m].axis("off")
            im = ax[l][m].contourf(
                v,
                r,
                Z,
                levels=np.linspace(-1, 1, 100),
                cmap=kwargs.get("cmap", "coolwarm"),
            )
            plot_data["amplitude"].append(Zs)

        cb_ax = plt.subplot(gs[:, -1])
        plt.subplots_adjust(right=0.8)
        cbar = fig.colorbar(im, cax=cb_ax)
        cbar.set_ticks(np.linspace(-1, 1, 9))
        fig.suptitle(
            "{}, $L={}$, $M={}$, spectral indexing = {}".format(
                basis.__class__.__name__, basis.L, basis.M, basis.spectral_indexing
            ),
            y=0.98,
            fontsize=title_fontsize,
        )
        _set_tight_layout(fig)
        if return_data:
            return fig, ax, plot_data

        return fig, ax


def plot_logo(save_path=None, **kwargs):
    """Plot the DESC logo.

    Parameters
    ----------
    save_path : str or path-like
        Path to save the figure to.
        File format is inferred from the filename (Default value = None)
    **kwargs : dict, optional
        Additional plot formatting parameters.
        options include ``'D_color'``, ``'D_color_rho'``, ``'D_color_theta'``,
        ``'E_color'``, ``'Scolor'``, ``'C_color'``, ``'BGcolor'``, ``'fig_width'``

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure being plotted to.
    ax : matplotlib.axes.Axes
        Axes being plotted to.

    Examples
    --------
    .. image:: ../../_static/images/plotting/plot_logo.png

    .. code-block:: python

        from desc.plotting import plot_logo
        plot_logo(save_path='../_static/images/plotting/plot_logo.png')

    """
    eq = np.array(
        [
            [0, 0, 0, 3.62287349e00, 0.00000000e00],
            [1, -1, 0, 0.00000000e00, 1.52398053e00],
            [1, 1, 0, 8.59865670e-01, 0.00000000e00],
            [2, -2, 0, 0.00000000e00, 1.46374759e-01],
            [2, 0, 0, -4.33377700e-01, 0.00000000e00],
            [2, 2, 0, 6.09609205e-01, 0.00000000e00],
            [3, -3, 0, 0.00000000e00, 2.13664220e-01],
            [3, -1, 0, 0.00000000e00, 1.29776568e-01],
            [3, 1, 0, -1.67706961e-01, 0.00000000e00],
            [3, 3, 0, 2.32179123e-01, 0.00000000e00],
            [4, -4, 0, 0.00000000e00, 3.30174283e-02],
            [4, -2, 0, 0.00000000e00, -5.80394864e-02],
            [4, 0, 0, -3.10228782e-02, 0.00000000e00],
            [4, 2, 0, -2.43905484e-03, 0.00000000e00],
            [4, 4, 0, 1.81292185e-01, 0.00000000e00],
            [5, -5, 0, 0.00000000e00, 5.37223061e-02],
            [5, -3, 0, 0.00000000e00, 2.65199520e-03],
            [5, -1, 0, 0.00000000e00, 1.63010516e-02],
            [5, 1, 0, 2.73622502e-02, 0.00000000e00],
            [5, 3, 0, -3.62812195e-02, 0.00000000e00],
            [5, 5, 0, 7.88069456e-02, 0.00000000e00],
            [6, -6, 0, 0.00000000e00, 3.50372526e-03],
            [6, -4, 0, 0.00000000e00, -1.82814700e-02],
            [6, -2, 0, 0.00000000e00, -1.62703504e-02],
            [6, 0, 0, 9.37285472e-03, 0.00000000e00],
            [6, 2, 0, 3.32793660e-03, 0.00000000e00],
            [6, 4, 0, -9.90606341e-03, 0.00000000e00],
            [6, 6, 0, 6.00068129e-02, 0.00000000e00],
            [7, -7, 0, 0.00000000e00, 1.28853330e-02],
            [7, -5, 0, 0.00000000e00, -2.28268526e-03],
            [7, -3, 0, 0.00000000e00, -1.04698799e-02],
            [7, -1, 0, 0.00000000e00, -5.15951605e-03],
            [7, 1, 0, 2.29082701e-02, 0.00000000e00],
            [7, 3, 0, -1.19760934e-02, 0.00000000e00],
            [7, 5, 0, -1.43418200e-02, 0.00000000e00],
            [7, 7, 0, 2.27668988e-02, 0.00000000e00],
            [8, -8, 0, 0.00000000e00, -2.53055423e-03],
            [8, -6, 0, 0.00000000e00, -7.15955981e-03],
            [8, -4, 0, 0.00000000e00, -6.54397837e-03],
            [8, -2, 0, 0.00000000e00, -4.08366006e-03],
            [8, 0, 0, 1.17264567e-02, 0.00000000e00],
            [8, 2, 0, -1.24364476e-04, 0.00000000e00],
            [8, 4, 0, -8.59425384e-03, 0.00000000e00],
            [8, 6, 0, -7.11934473e-03, 0.00000000e00],
            [8, 8, 0, 1.68974668e-02, 0.00000000e00],
        ]
    )

    onlyD = kwargs.get("onlyD", False)
    D_color = kwargs.get("D_color", "xkcd:neon purple")
    D_color_rho = kwargs.get("D_color_rho", "xkcd:neon pink")
    D_color_theta = kwargs.get("D_color_theta", "xkcd:neon pink")
    E_color = kwargs.get("E_color", "deepskyblue")
    Scolor = kwargs.get("Scolor", "deepskyblue")
    C_color = kwargs.get("C_color", "deepskyblue")
    BGcolor = kwargs.get("BGcolor", "clear")
    fig_width = kwargs.get("fig_width", 3)
    fig_height = fig_width / 2
    contour_lw_ratio = kwargs.get("contour_lw_ratio", 0.3)
    lw = fig_width**0.5

    transparent = False
    if BGcolor == "dark":
        BGcolor = "xkcd:charcoal grey"
    elif BGcolor == "light":
        BGcolor = "white"
    elif BGcolor == "clear":
        BGcolor = "white"
        transparent = True

    if onlyD:
        fig_width = fig_width / 2
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.axis("equal")
    ax.axis("off")
    ax.set_facecolor(BGcolor)
    fig.set_facecolor(BGcolor)
    if transparent:
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)

    bottom = 0
    top = 10
    Dleft = 0
    Dw = 8
    Dh = top - bottom + 2
    DX = Dleft + Dw / 2
    DY = (top - bottom) / 2
    Dright = Dleft + Dw

    Eleft = Dright + 0.5
    Eright = Eleft + 4

    Soffset = 1
    Sleft = Eright + 0.5
    Sw = 5
    Sright = Sleft + Sw

    Ctheta = np.linspace(np.pi / 4, 2 * np.pi - np.pi / 4, 1000)
    Cleft = Sright + 0.75
    Cw = 4
    Ch = 11
    Cx0 = Cleft + Cw / 2
    Cy0 = (top - bottom) / 2

    # D
    cR = eq[:, 3]
    cZ = eq[:, 4]
    zern_idx = eq[:, :3]
    ls, ms, ns = zern_idx.T
    axis_jacobi = zernike_radial_poly(0, ls, ms)
    R0 = axis_jacobi.dot(cR)
    Z0 = axis_jacobi.dot(cZ)

    nr = kwargs.get("nr", 5)
    nt = kwargs.get("nt", 8)
    Nr = 100
    Nt = 361
    rstep = Nr // nr
    tstep = Nt // nt
    r = np.linspace(0, 1, Nr)
    t = np.linspace(0, 2 * np.pi, Nt)
    r, t = np.meshgrid(r, t, indexing="ij")
    r = r.flatten()
    t = t.flatten()

    radial = zernike_radial_poly(r[:, np.newaxis], ls, ms)
    poloidal = fourier(t[:, np.newaxis], ms)
    zern = radial * poloidal
    bdry = poloidal

    R = zern.dot(cR).reshape((Nr, Nt))
    Z = zern.dot(cZ).reshape((Nr, Nt))
    bdryR = bdry.dot(cR)
    bdryZ = bdry.dot(cZ)

    R = (R - R0) / (R.max() - R.min()) * Dw + DX
    Z = (Z - Z0) / (Z.max() - Z.min()) * Dh + DY
    bdryR = (bdryR - R0) / (bdryR.max() - bdryR.min()) * Dw + DX
    bdryZ = (bdryZ - Z0) / (bdryZ.max() - bdryZ.min()) * Dh + DY

    # plot r contours
    ax.plot(
        R.T[:, ::rstep],
        Z.T[:, ::rstep],
        color=D_color_rho,
        lw=lw * contour_lw_ratio,
        ls="-",
    )
    # plot theta contours
    ax.plot(
        R[:, ::tstep],
        Z[:, ::tstep],
        color=D_color_theta,
        lw=lw * contour_lw_ratio,
        ls="-",
    )
    ax.plot(bdryR, bdryZ, color=D_color, lw=lw)

    if onlyD:
        if save_path is not None:
            fig.savefig(save_path, facecolor=fig.get_facecolor(), edgecolor="none")

        return fig, ax

    # E
    ax.plot([Eleft, Eleft + 1], [bottom, top], lw=lw, color=E_color, linestyle="-")
    ax.plot([Eleft, Eright], [bottom, bottom], lw=lw, color=E_color, linestyle="-")
    ax.plot(
        [Eleft + 1 / 2, Eright],
        [bottom + (top + bottom) / 2, bottom + (top + bottom) / 2],
        lw=lw,
        color=E_color,
        linestyle="-",
    )
    ax.plot([Eleft + 1, Eright], [top, top], lw=lw, color=E_color, linestyle="-")

    # S
    Sy = np.linspace(bottom, top + Soffset, 1000)
    Sx = Sw * np.cos(Sy * 3 / 2 * np.pi / (Sy.max() - Sy.min()) - np.pi) ** 2 + Sleft
    ax.plot(Sx, Sy[::-1] - Soffset / 2, lw=lw, color=Scolor, linestyle="-")

    # C
    Cx = Cw / 2 * np.cos(Ctheta) + Cx0
    Cy = Ch / 2 * np.sin(Ctheta) + Cy0
    ax.plot(Cx, Cy, lw=lw, color=C_color, linestyle="-")

    if save_path is not None:
        fig.savefig(save_path, facecolor=fig.get_facecolor(), edgecolor="none")

    return fig, ax
