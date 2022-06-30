from matplotlib import rcParams, cycler
import matplotlib
import numpy as np
import numbers
import tkinter
import re
from termcolor import colored
import warnings
from scipy.interpolate import Rbf
from scipy.integrate import solve_ivp

from desc.grid import Grid, LinearGrid
from desc.basis import zernike_radial_poly, fourier, DoubleFourierSeries
from desc.transform import Transform
from desc.compute import data_index
from desc.utils import flatten_list

__all__ = [
    "plot_1d",
    "plot_2d",
    "plot_3d",
    "plot_basis",
    "plot_boozer_modes",
    "plot_boozer_surface",
    "plot_coefficients",
    "plot_comparison",
    "plot_fsa",
    "plot_grid",
    "plot_logo",
    "plot_qs_error",
    "plot_section",
    "plot_surfaces",
]


colorblind_colors = [
    (0.0000, 0.4500, 0.7000),  # blue
    (0.8359, 0.3682, 0.0000),  # vermillion
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

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D


_axis_labels_rtz = [r"$\rho$", r"$\theta$", r"$\zeta$"]
_axis_labels_RPZ = [r"$R ~(\mathrm{m})$", r"$\phi$", r"$Z ~(\mathrm{m})$"]
_axis_labels_XYZ = [r"$X ~(\mathrm{m})$", r"$Y ~(\mathrm{m})$", r"$Z ~(\mathrm{m})$"]


def _format_ax(ax, is3d=False, rows=1, cols=1, figsize=None, equal=False):
    """Check type of ax argument. If ax is not a matplotlib AxesSubplot, initalize one.

    Parameters
    ----------
    ax : None or matplotlib AxesSubplot instance
        Axis to plot to.
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
                    "ax agument must be None or an axis instance or array of axes",
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
        "L": 1,
        "M": 1,
        "N": 1,
        "NFP": 1,
        "sym": False,
        "axis": True,
        "endpoint": True,
        "rho": None,
        "theta": None,
        "zeta": None,
    }
    for key in kwargs.keys():
        if key in grid_args.keys():
            grid_args[key] = kwargs[key]
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
    if np.unique(grid.nodes[:, 0]).size == 1:
        plot_axes.remove(0)
    if np.unique(grid.nodes[:, 1]).size == 1:
        plot_axes.remove(1)
    if np.unique(grid.nodes[:, 2]).size == 1:
        plot_axes.remove(2)

    return tuple(plot_axes)


def _compute(eq, name, grid, component=None):
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
    if name not in data_index:
        raise ValueError("Unrecognized value '{}'.".format(name))
    assert component in [
        None,
        "R",
        "phi",
        "Z",
    ], f"component must be one of [None, 'R', 'phi', 'Z'], got {component}"

    components = {
        "R": 0,
        "phi": 1,
        "Z": 2,
    }

    label = data_index[name]["label"]

    data = eq.compute(name, grid)[name]
    if data_index[name]["dim"] != 1:
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
    label = r"$" + label + "~(" + data_index[name]["units"] + ")$"

    return data.reshape((grid.M, grid.L, grid.N), order="F"), label


def plot_coefficients(eq, L=True, M=True, N=True, ax=None):
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

    fig, ax = _format_ax(ax, rows=1, cols=3)

    ax[0, 0].semilogy(
        np.sum(np.abs(eq.R_basis.modes[:, lmn]), axis=1), np.abs(eq.R_lmn), "bo"
    )
    ax[0, 1].semilogy(
        np.sum(np.abs(eq.Z_basis.modes[:, lmn]), axis=1), np.abs(eq.Z_lmn), "bo"
    )
    ax[0, 2].semilogy(
        np.sum(np.abs(eq.L_basis.modes[:, lmn]), axis=1), np.abs(eq.L_lmn), "bo"
    )

    ax[0, 0].set_xlabel(xlabel)
    ax[0, 1].set_xlabel(xlabel)
    ax[0, 2].set_xlabel(xlabel)

    ax[0, 0].set_title("$|R_{lmn}|$")
    ax[0, 1].set_title("$|Z_{lmn}|$")
    ax[0, 2].set_title("$|\\lambda_{lmn}|$")

    fig.set_tight_layout(True)
    return fig, ax


def plot_1d(eq, name, grid=None, log=False, ax=None, **kwargs):
    """Plot 1D profiles.

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
    ax : matplotlib AxesSubplot, optional
        Axis to plot on.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure being plotted to.
    ax : matplotlib.axes.Axes or ndarray of Axes
        Axes being plotted to.

    Examples
    --------

    .. image:: ../../_static/images/plotting/plot_1d.png

    .. code-block:: python

        from desc.plotting import plot_1d
        plot_1d(eq, 'p')

    """
    if grid is None:
        grid_kwargs = {"L": 100, "NFP": eq.NFP}
        grid = _get_grid(**grid_kwargs)
    plot_axes = _get_plot_axes(grid)
    if len(plot_axes) != 1:
        return ValueError(colored("Grid must be 1D", "red"))

    data, label = _compute(eq, name, grid, kwargs.get("component", None))
    fig, ax = _format_ax(ax, figsize=kwargs.get("figsize", (4, 4)))

    # reshape data to 1D
    data = data.flatten()

    if log:
        data = np.abs(data)  # ensure data is positive for log plot
        ax.semilogy(grid.nodes[:, plot_axes[0]], data, label=kwargs.get("label", None))
    else:
        ax.plot(grid.nodes[:, plot_axes[0]], data, label=kwargs.get("label", None))

    ax.set_xlabel(_axis_labels_rtz[plot_axes[0]])
    ax.set_ylabel(label)
    fig.set_tight_layout(True)
    return fig, ax


def plot_2d(eq, name, grid=None, log=False, norm_F=False, ax=None, **kwargs):
    """Plot 2D cross-sections.

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

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure being plotted to.
    ax : matplotlib.axes.Axes or ndarray of Axes
        Axes being plotted to.

    Examples
    --------

    .. image:: ../../_static/images/plotting/plot_2d.png

    .. code-block:: python

        from desc.plotting import plot_2d
        plot_2d(eq, 'sqrt(g)')

    """
    if grid is None:
        grid_kwargs = {"M": 33, "N": 33, "NFP": eq.NFP, "axis": False}
        grid = _get_grid(**grid_kwargs)
    plot_axes = _get_plot_axes(grid)
    if len(plot_axes) != 2:
        return ValueError(colored("Grid must be 2D", "red"))

    data, label = _compute(eq, name, grid, kwargs.get("component", None))
    fig, ax = _format_ax(ax, figsize=kwargs.get("figsize", (4, 4)))
    divider = make_axes_locatable(ax)

    if norm_F:
        if name != "|F|":
            return ValueError(colored("Can only normalize |F|.", "red"))
        else:
            if (
                np.max(abs(eq.p_l)) <= np.finfo(eq.p_l.dtype).eps
            ):  # normalize vacuum force by B pressure gradient
                norm_name = "|grad(|B|^2)|/2mu0"
            else:  # normalize force balance with pressure by gradient of pressure
                norm_name = "|grad(p)|"
            norm_data, _ = _compute(eq, norm_name, grid)
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
            contourf_kwargs["levels"] = kwargs.get("levels", np.logspace(-6, 0, 7))
        else:
            logmin = max(np.floor(np.nanmin(np.log10(data))).astype(int), -16)
            logmax = np.ceil(np.nanmax(np.log10(data))).astype(int)
            contourf_kwargs["levels"] = kwargs.get(
                "levels", np.logspace(logmin, logmax, logmax - logmin + 1)
            )
    else:
        contourf_kwargs["norm"] = matplotlib.colors.Normalize()
        contourf_kwargs["levels"] = kwargs.get(
            "levels", np.linspace(np.nanmin(data), np.nanmax(data), 100)
        )
    contourf_kwargs["cmap"] = kwargs.get("cmap", "jet")
    contourf_kwargs["extend"] = "both"

    cax_kwargs = {"size": "5%", "pad": 0.05}

    xx = (
        grid.nodes[:, plot_axes[1]]
        .reshape((grid.M, grid.L, grid.N), order="F")
        .squeeze()
    )
    yy = (
        grid.nodes[:, plot_axes[0]]
        .reshape((grid.M, grid.L, grid.N), order="F")
        .squeeze()
    )

    im = ax.contourf(xx, yy, data, **contourf_kwargs)
    cax = divider.append_axes("right", **cax_kwargs)
    cbar = fig.colorbar(im, cax=cax)
    cbar.update_ticks()

    ax.set_xlabel(_axis_labels_rtz[plot_axes[1]])
    ax.set_ylabel(_axis_labels_rtz[plot_axes[0]])
    ax.set_title(label)
    if norm_F:
        ax.set_title(
            "%s / %s"
            % (
                "$" + data_index[name]["label"] + "$",
                "$" + data_index[norm_name]["label"] + "$",
            )
        )
    fig.set_tight_layout(True)
    return fig, ax


def plot_3d(eq, name, grid=None, log=False, all_field_periods=True, ax=None, **kwargs):
    """Plot 3D surfaces.

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
    all_field_periods : bool, optional
        Whether to plot full torus or one field period. Ignored if grid is specified.
    ax : matplotlib AxesSubplot, optional
        Axis to plot on.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure being plotted to.
    ax : matplotlib.axes.Axes or ndarray of Axes
        Axes being plotted to.

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
        fig, ax = plot_3d(eq, "|F|", log=True, grid=grid)

    """
    nfp = 1 if all_field_periods else eq.NFP
    if grid is None:
        grid_kwargs = {"M": 33, "N": int(33 * eq.NFP), "NFP": nfp}
        grid = _get_grid(**grid_kwargs)
    plot_axes = _get_plot_axes(grid)
    if len(plot_axes) != 2:
        return ValueError(colored("Grid must be 2D", "red"))

    data, label = _compute(eq, name, grid, kwargs.get("component", None))
    fig, ax = _format_ax(ax, is3d=True, figsize=kwargs.get("figsize", None))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coords = eq.compute("X", grid)
    X = coords["X"].reshape((grid.M, grid.L, grid.N), order="F")
    Y = coords["Y"].reshape((grid.M, grid.L, grid.N), order="F")
    Z = coords["Z"].reshape((grid.M, grid.L, grid.N), order="F")

    if 0 in plot_axes:
        if 1 in plot_axes:  # rho & theta
            data = data[:, :, 0]
            X = X[:, :, 0]
            Y = Y[:, :, 0]
            Z = Z[:, :, 0]
        else:  # rho & zeta
            data = data[0, :, :].T
            X = X[0, :, :].T
            Y = Y[0, :, :].T
            Z = Z[0, :, :].T
    else:  # theta & zeta
        data = data[:, 0, :].T
        X = X[:, 0, :].T
        Y = Y[:, 0, :].T
        Z = Z[:, 0, :].T

    if log:
        data = np.abs(data)  # ensure data is positive for log plot
        minn, maxx = data.min().min(), data.max().max()
        norm = matplotlib.colors.LogNorm(vmin=minn, vmax=maxx)
    else:
        minn, maxx = data.min().min(), data.max().max()
        norm = matplotlib.colors.Normalize(vmin=minn, vmax=maxx)
    m = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=norm)
    m.set_array([])
    alpha = kwargs.get("alpha", 1)

    ax.plot_surface(
        X,
        Y,
        Z,
        cmap="jet",
        facecolors=plt.cm.jet(norm(data)),
        vmin=minn,
        vmax=maxx,
        rstride=1,
        cstride=1,
        alpha=alpha,
    )
    fig.colorbar(m)

    ax.set_xlabel(_axis_labels_XYZ[0])
    ax.set_ylabel(_axis_labels_XYZ[1])
    ax.set_zlabel(_axis_labels_XYZ[2])
    ax.set_title(label)
    fig.set_tight_layout(True)

    # need this stuff to make all the axes equal, ax.axis('equal') doesnt work for 3d
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    return fig, ax


def plot_fsa(
    eq,
    name,
    log=False,
    L=20,
    M=None,
    N=None,
    rho=None,
    ax=None,
    **kwargs,
):
    """Plot flux surface averaged quantities.

    Parameters
    ----------
    eq : Equilibrium
        Object from which to plot.
    name : str
        Name of variable to plot.
    log : bool, optional
        Whether to use a log scale.
    L : int, optional
        Number of flux surfaces to evaluate at. Only used if rho=None.
    M : int, optional
        Number of poloidal nodes used in flux surface average. Default is 2*eq.M_grid+1.
    N : int, optional
        Number of toroidal nodes used in flux surface average. Default is 2*eq.N_grid+1.
    rho : ndarray, optional
        Radial coordinates of the flux surfaces to evaluate at.
    ax : matplotlib AxesSubplot, optional
        Axis to plot on.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure being plotted to.
    ax : matplotlib.axes.Axes or ndarray of Axes
        Axes being plotted to.

    Examples
    --------

    .. image:: ../../_static/images/plotting/plot_fsa.png

    .. code-block:: python

        from desc.plotting import plot_fsa
        fig, ax = plot_fsa(eq, "B_theta")

    """
    if rho is None:
        rho = np.linspace(1, 0, num=L, endpoint=False)
    if M is None:
        M = 2 * eq.M_grid + 1
    if N is None:
        N = 2 * eq.N_grid + 1

    fig, ax = _format_ax(ax, figsize=kwargs.get("figsize", (4, 4)))

    values = np.array([])
    for i, r in enumerate(rho):
        grid = LinearGrid(M=M, N=N, NFP=1, rho=r)
        g, _ = _compute(eq, "sqrt(g)", grid)
        data, label = _compute(eq, name, grid, kwargs.get("component", None))
        values = np.append(values, np.mean(data * g) / np.mean(g))

    if log:
        values = np.abs(values)  # ensure data is positive for log plot
        ax.semilogy(rho, values, label=kwargs.get("label", None))
    else:
        ax.plot(rho, values, label=kwargs.get("label", None))

    label = label.split("~")
    label = r"$\langle " + label[0][1:] + r" \rangle~" + "~".join(label[1:])

    ax.set_xlabel(_axis_labels_rtz[0])
    ax.set_ylabel(label)
    fig.set_tight_layout(True)
    return fig, ax


def plot_section(eq, name, grid=None, log=False, norm_F=False, ax=None, **kwargs):
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

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure being plotted to.
    ax : matplotlib.axes.Axes or ndarray of Axes
        Axes being plotted to.

    Examples
    --------

    .. image:: ../../_static/images/plotting/plot_section.png

    .. code-block:: python

        from desc.plotting import plot_section
        fig, ax = plot_section(eq, "J^rho")

    """
    if grid is None:
        if eq.N == 0:
            nzeta = int(kwargs.get("nzeta", 1))
        else:
            nzeta = int(kwargs.get("nzeta", 6))
        nfp = eq.NFP
        grid_kwargs = {
            "L": 25,
            "NFP": nfp,
            "axis": False,
            "theta": np.linspace(0, 2 * np.pi, 91, endpoint=True),
            "zeta": np.linspace(0, 2 * np.pi / nfp, nzeta, endpoint=False),
        }
        grid = _get_grid(**grid_kwargs)
        zeta = np.unique(grid.nodes[:, 2])

    else:
        zeta = np.unique(grid.nodes[:, 2])
        nzeta = zeta.size
    rows = np.floor(np.sqrt(nzeta)).astype(int)
    cols = np.ceil(nzeta / rows).astype(int)

    data, label = _compute(eq, name, grid, kwargs.get("component", None))
    if norm_F:
        if name != "|F|":
            return ValueError(colored("Can only normalize |F|.", "red"))
        else:
            if (
                np.max(abs(eq.p_l)) <= np.finfo(eq.p_l.dtype).eps
            ):  # normalize vacuum force by B pressure gradient
                norm_name = "|grad(|B|^2)|/2mu0"
            else:  # normalize force balance with pressure by gradient of pressure
                norm_name = "|grad(p)|"
            norm_data, _ = _compute(eq, norm_name, grid)
            data = data / np.nanmean(np.abs(norm_data))  # normalize

    figw = 5 * cols
    figh = 5 * rows
    fig, ax = _format_ax(
        ax,
        rows=rows,
        cols=cols,
        figsize=kwargs.get("figsize", (figw, figh)),
        equal=True,
    )
    ax = np.atleast_1d(ax).flatten()

    coords = eq.compute("R", grid)
    R = coords["R"].reshape((grid.M, grid.L, grid.N), order="F")
    Z = coords["Z"].reshape((grid.M, grid.L, grid.N), order="F")

    contourf_kwargs = {}
    if log:
        data = np.abs(data)  # ensure data is positive for log plot
        contourf_kwargs["norm"] = matplotlib.colors.LogNorm()
        if norm_F:
            contourf_kwargs["levels"] = kwargs.get("levels", np.logspace(-6, 0, 7))
        else:
            logmin = np.floor(np.nanmin(np.log10(data))).astype(int)
            logmax = np.ceil(np.nanmax(np.log10(data))).astype(int)
            contourf_kwargs["levels"] = kwargs.get(
                "levels", np.logspace(logmin, logmax, logmax - logmin + 1)
            )
    else:
        contourf_kwargs["norm"] = matplotlib.colors.Normalize()
        contourf_kwargs["levels"] = kwargs.get(
            "levels", np.linspace(data.min(), data.max(), 100)
        )
    contourf_kwargs["cmap"] = kwargs.get("cmap", "jet")
    contourf_kwargs["extend"] = "both"

    cax_kwargs = {"size": "5%", "pad": 0.05}

    for i in range(nzeta):
        divider = make_axes_locatable(ax[i])

        cntr = ax[i].contourf(R[:, :, i], Z[:, :, i], data[:, :, i], **contourf_kwargs)
        cax = divider.append_axes("right", **cax_kwargs)
        cbar = fig.colorbar(cntr, cax=cax)
        cbar.update_ticks()

        ax[i].set_xlabel(_axis_labels_RPZ[0])
        ax[i].set_ylabel(_axis_labels_RPZ[2])
        ax[i].tick_params(labelbottom=True, labelleft=True)
        ax[i].set_title(
            "$"
            + data_index[name]["label"]
            + "$ ($"
            + data_index[name]["units"]
            + "$)"
            + ", $\\zeta \\cdot NFP/2\\pi = {:.3f}$".format(
                eq.NFP * zeta[i] / (2 * np.pi)
            )
        )
        if norm_F:
            ax[i].set_title(
                "%s / %s, %s"
                % (
                    "$" + data_index[name]["label"] + "$",
                    "$" + data_index[norm_name]["label"] + "$",
                    "$\\zeta \\cdot NFP/2\\pi = {:.3f}$".format(
                        eq.NFP * zeta[i] / (2 * np.pi)
                    ),
                )
            )
    fig.set_tight_layout(True)
    return fig, ax


def plot_surfaces(eq, rho=8, theta=8, zeta=None, ax=None, **kwargs):
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
    zeta : int or array-like or None
        Values of zeta to plot contours at.
        If an integer, plot that many contours linearly spaced in (0,2pi).
        Default is 1 contour for axisymmetric equilibria or 6 for non-axisymmetry.
    ax : matplotlib AxesSubplot, optional
        Axis to plot on.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure being plotted to.
    ax : matplotlib.axes.Axes or ndarray of Axes
        Axes being plotted to.

    Examples
    --------

    .. image:: ../../_static/images/plotting/plot_surfaces.png

    .. code-block:: python

        from desc.plotting import plot_surfaces
        fig, ax = plot_surfaces(eq)

    """
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
    if len(kwargs):
        raise ValueError(
            f"plot surfaces got unexpected keyword argument: {kwargs.keys()}"
        )

    nfp = eq.NFP
    if isinstance(rho, numbers.Integral):
        rho = np.linspace(0, 1, rho + 1)  # offset to ignore axis
    else:
        rho = np.atleast_1d(rho)
    if isinstance(theta, numbers.Integral):
        theta = np.linspace(0, 2 * np.pi, theta, endpoint=False)
    else:
        theta = np.atleast_1d(theta)
    if isinstance(zeta, numbers.Integral):
        zeta = np.linspace(0, 2 * np.pi / nfp, zeta)
    elif zeta is None:
        if eq.N == 0:
            zeta = np.array([0])
        else:
            zeta = np.linspace(0, 2 * np.pi / nfp, 6, endpoint=False)
    else:
        zeta = np.atleast_1d(zeta)
    nzeta = len(zeta)

    grid_kwargs = {
        "rho": rho,
        "NFP": nfp,
        "theta": np.linspace(0, 2 * np.pi, NT, endpoint=True),
        "zeta": zeta,
    }
    r_grid = _get_grid(**grid_kwargs)
    grid_kwargs = {
        "rho": np.linspace(0, 1, NR),
        "NFP": nfp,
        "theta": theta,
        "zeta": zeta,
    }
    t_grid = _get_grid(**grid_kwargs)

    # Note: theta* (also known as vartheta) is the poloidal straight field-line anlge in
    # PEST-like flux coordinates

    v_grid = Grid(eq.compute_theta_coords(t_grid.nodes))
    rows = np.floor(np.sqrt(nzeta)).astype(int)
    cols = np.ceil(nzeta / rows).astype(int)

    # rho contours
    r_coords = eq.compute("R", r_grid)
    Rr = r_coords["R"].reshape((r_grid.M, r_grid.L, r_grid.N), order="F")
    Zr = r_coords["Z"].reshape((r_grid.M, r_grid.L, r_grid.N), order="F")

    # vartheta contours
    v_coords = eq.compute("R", v_grid)
    Rv = v_coords["R"].reshape((t_grid.M, t_grid.L, t_grid.N), order="F")
    Zv = v_coords["Z"].reshape((t_grid.M, t_grid.L, t_grid.N), order="F")

    figw = 4 * cols
    figh = 5 * rows
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

    for i in range(nzeta):
        ax[i].plot(
            Rv[:, :, i].T,
            Zv[:, :, i].T,
            color=theta_color,
            linestyle=theta_ls,
            lw=theta_lw,
        )
        ax[i].plot(
            Rr[:, :, i],
            Zr[:, :, i],
            color=rho_color,
            linestyle=rho_ls,
            lw=rho_lw,
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

        ax[i].set_xlabel(_axis_labels_RPZ[0])
        ax[i].set_ylabel(_axis_labels_RPZ[2])
        ax[i].tick_params(labelbottom=True, labelleft=True)
        ax[i].set_title(
            "$\\zeta \\cdot NFP/2\\pi = {:.3f}$".format(nfp * zeta[i] / (2 * np.pi))
        )
    fig.set_tight_layout(True)
    return fig, ax


def plot_comparison(
    eqs,
    rho=8,
    theta=8,
    zeta=None,
    ax=None,
    cmap="rainbow",
    colors=None,
    lws=None,
    linestyles=None,
    labels=None,
    **kwargs,
):
    """Plot comparison between flux surfaces of multiple equilibria.

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
    zeta : int or array-like or None
        Values of zeta to plot contours at.
        If an integer, plot that many contours linearly spaced in (0,2pi).
        Default is 1 contour for axisymmetric equilibria or 6 for non-axisymmetry.
    ax : matplotlib AxesSubplot, optional
        Axis to plot on.
    cmap : str or matplotlib ColorMap
        Colormap to use for plotting, discretized into len(eqs) colors.
    colors : array-like
        Array the same length as eqs of colors to use for each equilibrium.
        Overrides `cmap`.
    lws : array-like
        Array the same length as eqs of line widths to use for each equilibrium
    linestyles : array-like
        Array the same length as eqs of linestyles to use for each equilibrium.
    labels : array-like
        Array the same length as eqs of labels to apply to each equilibrium.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure being plotted to.
    ax : matplotlib.axes.Axes or ndarray of Axes
        Axes being plotted to.

    Examples
    --------

    .. image:: ../../_static/images/plotting/plot_comparison.png

    .. code-block:: python

        from desc.plotting import plot_comparison
        fig, ax = plot_comparison(eqs=[eqf[0],eqf[1],eqf[2]],labels=['Axisymmetric w/o pressure','Axisymmetric w/ pressure','Nonaxisymmetric w/ pressure'])

    """
    figsize = kwargs.pop("figsize", None)
    neq = len(eqs)
    if colors is None:
        colors = matplotlib.cm.get_cmap(cmap, neq)(np.linspace(0, 1, neq))
    if lws is None:
        lws = [1 for i in range(neq)]
    if linestyles is None:
        linestyles = ["-" for i in range(neq)]
    if labels is None:
        labels = [str(i) for i in range(neq)]
    N = np.max([eq.N for eq in eqs])
    nfp = eqs[0].NFP
    if isinstance(zeta, numbers.Integral):
        zeta = np.linspace(0, 2 * np.pi / nfp, zeta)
    elif zeta is None:
        if N == 0:
            zeta = np.array([0])
        else:
            zeta = np.linspace(0, 2 * np.pi / nfp, 6, endpoint=False)
    else:
        zeta = np.atleast_1d(zeta)
    nzeta = len(zeta)
    rows = np.floor(np.sqrt(nzeta)).astype(int)
    cols = np.ceil(nzeta / rows).astype(int)

    figw = 4 * cols
    figh = 5 * rows
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
    for i, eq in enumerate(eqs):
        fig, ax = plot_surfaces(
            eq,
            rho,
            theta,
            zeta,
            ax,
            theta_color=colors[i % len(colors)],
            theta_ls=linestyles[i % len(linestyles)],
            theta_lw=lws[i % len(lws)],
            rho_color=colors[i % len(colors)],
            rho_ls=linestyles[i % len(linestyles)],
            rho_lw=lws[i % len(lws)],
            lcfs_color=colors[i % len(colors)],
            lcfs_ls=linestyles[i % len(linestyles)],
            lcfs_lw=lws[i % len(lws)],
            axis_color=colors[i % len(colors)],
            axis_alpha=0,
            axis_marker="o",
            axis_size=0,
            label=labels[i % len(labels)],
        )
    if any(labels) and kwargs.get("legend", True):
        fig.legend(**kwargs.get("legend_kw", {}))
    return fig, ax


def plot_coils(coils, grid=None, ax=None, **kwargs):
    """Create 3D plot of coil geometry

    Parameters
    ----------
    coils : Coil, CoilSet
        Coil or coils to plot
    grid : Grid, optional
        Grid to use for evaluating geometry
    ax : matplotlib AxesSubplot, optional
        Axis to plot on

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure being plotted to
    ax : matplotlib.axes.Axes or ndarray of Axes
        Axes being plotted to
    """

    figsize = kwargs.pop("figsize", None)
    lw = kwargs.pop("lw", 2)
    ls = kwargs.pop("ls", "-")
    color = kwargs.pop("color", "current")
    color = kwargs.pop("c", color)
    cbar = False
    if color == "current":
        cbar = True
        cmap = matplotlib.cm.get_cmap(kwargs.pop("cmap", "Spectral"))
        currents = flatten_list(coils.current)
        norm = matplotlib.colors.Normalize(vmin=np.min(currents), vmax=np.max(currents))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        color = [cmap(norm(cur)) for cur in currents]
    if not isinstance(lw, (list, tuple)):
        lw = [lw]
    if not isinstance(ls, (list, tuple)):
        ls = [ls]
    if not isinstance(color, (list, tuple)):
        color = [color]
    fig, ax = _format_ax(ax, True, figsize=figsize)
    if grid is None:
        grid_kwargs = {
            "zeta": np.linspace(0, 2 * np.pi, 50),
        }
        grid = _get_grid(**grid_kwargs)

    def flatten_coils(coilset):
        if hasattr(coilset, "__len__"):
            return [a for i in coilset for a in flatten_coils(i)]
        else:
            return [coilset]

    coils_list = flatten_coils(coils)

    for i, coil in enumerate(coils_list):
        x, y, z = coil.compute_coordinates(grid=grid, basis="xyz").T
        ax.plot(
            x, y, z, lw=lw[i % len(lw)], ls=ls[i % len(ls)], c=color[i % len(color)]
        )

    if cbar:
        cbar = fig.colorbar(sm)
        cbar.set_label(r"$\mathrm{Current} ~(\mathrm{A})$")
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence we call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    ax.set_xlabel(_axis_labels_XYZ[0])
    ax.set_ylabel(_axis_labels_XYZ[1])
    ax.set_zlabel(_axis_labels_XYZ[2])

    return fig, ax


def plot_boozer_modes(eq, log=True, B0=True, num_modes=10, rho=None, ax=None, **kwargs):
    """Plot Fourier harmonics of :math:`|B|` in Boozer coordinates.

    Parameters
    ----------
    eq : Equilibrium
        Object from which to plot.
    log : bool, optional
        Whether to use a log scale.
    B0 : bool, optional
        Whether to include the m=n=0 mode.
    num_modes : int, optional
        How many modes to include. Default (-1) is all.
    rho : int or ndarray, optional
        Radial coordinates of the flux surfaces to evaluate at,
        or number of surfaces in (0,1]
    ax : matplotlib AxesSubplot, optional
        Axis to plot on.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure being plotted to.
    ax : matplotlib.axes.Axes or ndarray of Axes
        Axes being plotted to.

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
    ds = []
    B_mn = np.array([[]])
    linestyle = kwargs.get("linestyle", "-")
    for i, r in enumerate(rho):
        grid = LinearGrid(M=6 * eq.M + 1, N=6 * eq.N + 1, NFP=eq.NFP, rho=r)
        data = eq.compute("|B|_mn", grid)
        ds.append(data)
        b_mn = np.atleast_2d(data["|B|_mn"])
        B_mn = np.vstack((B_mn, b_mn)) if B_mn.size else b_mn
    idx = np.argsort(np.abs(B_mn[0, :]))
    if num_modes == -1:
        idx = idx[-1::-1]
    else:
        idx = idx[-1 : -num_modes - 1 : -1]
    B_mn = B_mn[:, idx]
    modes = data["B modes"][idx, :]

    fig, ax = _format_ax(ax)
    for i in range(modes.shape[0]):
        M = modes[i, 1]
        N = modes[i, 2]
        if (M, N) == (0, 0) and B0 is False:
            continue
        if log is True:
            ax.semilogy(
                rho,
                np.abs(B_mn[:, i]),
                label="M={}, N={}".format(M, N),
                linestyle=linestyle,
            )
        else:
            ax.plot(
                rho,
                B_mn[:, i],
                "-",
                label="M={}, N={}".format(M, N),
                linestyle=linestyle,
            )

    ax.set_xlabel(_axis_labels_rtz[0])
    ax.set_ylabel(r"$B_{M,N}$ in Boozer coordinates $(T)$")
    fig.legend(loc="center right")

    fig.set_tight_layout(True)
    return fig, ax


def plot_boozer_surface(
    eq, grid_compute=None, grid_plot=None, fill=True, ncontours=100, ax=None, **kwargs
):
    """Plot :math:`|B|` on a surface vs the Boozer poloidal and toroidal angles.

    Parameters
    ----------
    eq : Equilibrium
        Object from which to plot.
    grid_compute : Grid, optional
        grid to use for computing boozer spectrum
    grid_plot : Grid, optional
        grid to plot on
    fill : bool, optional
        Whether the contours are filled, i.e. whether to use `contourf` or `contour`.
    ncontours : int, optional
        Number of contours to plot.
    ax : matplotlib AxesSubplot, optional
        Axis to plot on.

    Returns
    -------
    fig : matplotlib.figure.Figure
        figure being plotted to
    ax : matplotlib.axes.Axes or ndarray of Axes
        axes being plotted to

    Examples
    --------

    .. image:: ../../_static/images/plotting/plot_boozer_surface.png

    .. code-block:: python

        from desc.plotting import plot_boozer_surface
        fig, ax = plot_boozer_surface(eq)

    """
    if grid_compute is None:
        grid_kwargs = {
            "M": 6 * eq.M + 1,
            "N": 6 * eq.N + 1,
            "NFP": eq.NFP,
            "endpoint": False,
        }
        grid_compute = _get_grid(**grid_kwargs)
    if grid_plot is None:
        grid_kwargs = {"M": 100, "N": 100, "NFP": eq.NFP, "endpoint": True}
        grid_plot = _get_grid(**grid_kwargs)

    data = eq.compute("|B|_mn", grid_compute)
    B_transform = Transform(
        grid_plot,
        DoubleFourierSeries(M=2 * eq.M, N=2 * eq.N, sym=eq.R_basis.sym, NFP=eq.NFP),
    )
    data = B_transform.transform(data["|B|_mn"])
    data = data.reshape((grid_plot.M, grid_plot.N), order="F")

    fig, ax = _format_ax(ax, figsize=kwargs.get("figsize", (4, 4)))
    divider = make_axes_locatable(ax)

    contourf_kwargs = {}
    contourf_kwargs["norm"] = matplotlib.colors.Normalize()
    contourf_kwargs["levels"] = kwargs.get(
        "levels", np.linspace(np.nanmin(data), np.nanmax(data), ncontours)
    )
    contourf_kwargs["cmap"] = kwargs.get("cmap", "jet")
    contourf_kwargs["extend"] = "both"

    cax_kwargs = {"size": "5%", "pad": 0.05}

    xx = grid_plot.nodes[:, 2].reshape((grid_plot.M, grid_plot.N), order="F").squeeze()
    yy = grid_plot.nodes[:, 1].reshape((grid_plot.M, grid_plot.N), order="F").squeeze()

    if fill:
        im = ax.contourf(xx, yy, data, **contourf_kwargs)
    else:
        im = ax.contour(xx, yy, data, **contourf_kwargs)
    cax = divider.append_axes("right", **cax_kwargs)
    cbar = fig.colorbar(im, cax=cax)
    cbar.update_ticks()

    ax.set_xlabel(r"$\zeta_{Boozer}$")
    ax.set_ylabel(r"$\theta_{Boozer}$")
    ax.set_title(r"$|\mathbf{B}|~(T)$")

    fig.set_tight_layout(True)
    return fig, ax


def plot_qs_error(
    eq,
    log=True,
    fB=True,
    fC=True,
    fT=True,
    helicity=(1, 0),
    rho=None,
    ax=None,
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

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure being plotted to.
    ax : matplotlib.axes.Axes or ndarray of Axes
        Axes being plotted to.

    Examples
    --------

    .. image:: ../../_static/images/plotting/plot_qs_error.png

    .. code-block:: python

        from desc.plotting import plot_qs_error
        fig, ax = plot_qs_error(eq, helicity=(1, eq.NFP), log=True)

    """
    if rho is None:
        rho = np.linspace(1, 0, num=20, endpoint=False)
    elif np.isscalar(rho) and rho > 1:
        rho = np.linspace(1, 0, num=rho, endpoint=False)

    fig, ax = _format_ax(ax)

    ls = kwargs.get("ls", ["-", "-", "-"])
    colors = kwargs.get("colors", ["r", "b", "g"])
    markers = kwargs.get("markers", ["o", "o", "o"])

    data = eq.compute("R0")
    data = eq.compute("|B|", data=data)
    R0 = data["R0"]
    B0 = np.mean(data["|B|"] * data["sqrt(g)"]) / np.mean(data["sqrt(g)"])

    data = None
    f_B = np.array([])
    f_C = np.array([])
    f_T = np.array([])
    for i, r in enumerate(rho):
        grid = LinearGrid(M=2 * eq.M_grid + 1, N=2 * eq.N_grid + 1, NFP=eq.NFP, rho=r)
        if fB:
            data = eq.compute("|B|_mn", grid, data)
            modes = data["B modes"]
            idx = np.where((modes[1, :] * helicity[1] != modes[2, :] * helicity[0]))[0]
            f_b = np.sqrt(np.sum(data["|B|_mn"][idx] ** 2)) / np.sqrt(
                np.sum(data["|B|_mn"] ** 2)
            )
            f_B = np.append(f_B, f_b)
        if fC:
            data = eq.compute("f_C", grid, data)
            f_c = (
                np.mean(np.abs(data["f_C"]) * data["sqrt(g)"])
                / np.mean(data["sqrt(g)"])
                / B0 ** 3
            )
            f_C = np.append(f_C, f_c)
        if fT:
            data = eq.compute("f_T", grid, data)
            f_t = (
                np.mean(np.abs(data["f_T"]) * data["sqrt(g)"])
                / np.mean(data["sqrt(g)"])
                * R0 ** 2
                / B0 ** 4
            )
            f_T = np.append(f_T, f_t)

    if log is True:
        if fB:
            ax.semilogy(
                rho,
                f_B,
                ls=ls[0 % len(ls)],
                c=colors[0 % len(colors)],
                marker=markers[0 % len(markers)],
                label=r"$\hat{f}_B$",
            )
        if fC:
            ax.semilogy(
                rho,
                f_C,
                ls=ls[1 % len(ls)],
                c=colors[1 % len(colors)],
                marker=markers[1 % len(markers)],
                label=r"$\hat{f}_C$",
            )
        if fT:
            ax.semilogy(
                rho,
                f_T,
                ls=ls[2 % len(ls)],
                c=colors[2 % len(colors)],
                marker=markers[2 % len(markers)],
                label=r"$\hat{f}_T$",
            )
    else:
        if fB:
            ax.plot(
                rho,
                f_B,
                ls=ls[0 % len(ls)],
                c=colors[0 % len(colors)],
                marker=markers[0 % len(markers)],
                label=r"$\hat{f}_B$",
            )
        if fC:
            ax.plot(
                rho,
                f_C,
                ls=ls[1 % len(ls)],
                c=colors[1 % len(colors)],
                marker=markers[1 % len(markers)],
                label=r"$\hat{f}_C$",
            )
        if fT:
            ax.plot(
                rho,
                f_T,
                ls=ls[2 % len(ls)],
                c=colors[2 % len(colors)],
                marker=markers[2 % len(markers)],
                label=r"$\hat{f}_T$",
            )

    ax.set_xlabel(_axis_labels_rtz[0])
    if kwargs.get("legend", True):
        fig.legend(**kwargs.get("legend_kwargs", {"loc": "center right"}))

    fig.set_tight_layout(True)
    return fig, ax


def plot_grid(grid, **kwargs):
    """Plot the location of collocation nodes on the zeta=0 plane.

    Parameters
    ----------
    grid : Grid
        Grid to plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure being plotted to.
    ax : matplotlib.axes.Axes or ndarray of Axes
        Axes being plotted to.

    Examples
    --------

    .. image:: ../../_static/images/plotting/plot_grid.png

    .. code-block:: python

        from desc.plotting import plot_grid
        from desc.grid import ConcentricGrid
        grid = ConcentricGrid(L=20, M=10, N=1, node_pattern="jacobi")
        fig, ax = plot_grid(grid)

    """
    fig = plt.figure(figsize=kwargs.get("figsize", (4, 4)))
    ax = plt.subplot(projection="polar")

    # node locations
    nodes = grid.nodes[np.where(grid.nodes[:, 2] == 0)]
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
            r"$\frac{4\pi}{4}$",
            r"$\frac{3\pi}{2}$",
            r"$2\pi$",
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
        )
    fig.set_tight_layout(True)
    return fig, ax


def plot_basis(basis, **kwargs):
    """Plot basis functions.

    Parameters
    ----------
    basis : Basis
        basis to plot

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure being plotted to.
    ax : matplotlib.axes.Axes, ndarray of axes, or dict of axes
        Axes used for plotting. A single axis is used for 1d basis functions,
        2d or 3d bases return an ndarray or dict of axes.

    Examples
    --------

    .. image:: ../../_static/images/plotting/plot_basis.png

    .. code-block:: python

        from desc.plotting import plot_basis
        from desc.basis import DoubleFourierSeries
        basis = DoubleFourierSeries(M=3, N=2)
        fig, ax = plot_basis(basis)

    """
    if basis.__class__.__name__ == "PowerSeries":
        lmax = abs(basis.modes[:, 0]).max()
        grid = LinearGrid(100, 1, 1, endpoint=True)
        r = grid.nodes[:, 0]
        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (6, 4)))

        f = basis.evaluate(grid.nodes)
        for fi, l in zip(f.T, basis.modes[:, 0]):
            ax.plot(r, fi, label="$l={:d}$".format(int(l)))
        ax.set_xlabel("$\\rho$")
        ax.set_ylabel("$f_l(\\rho)$")
        ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_title("{}, $L={}$".format(basis.__class__.__name__, basis.L))
        fig.set_tight_layout(True)
        return fig, ax

    elif basis.__class__.__name__ == "FourierSeries":
        nmax = abs(basis.modes[:, 2]).max()
        grid = LinearGrid(1, 1, 100, NFP=basis.NFP, endpoint=True)
        z = grid.nodes[:, 2]
        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (6, 4)))

        f = basis.evaluate(grid.nodes)
        for fi, n in zip(f.T, basis.modes[:, 2]):
            ax.plot(z, fi, label="$n={:d}$".format(int(n)))
        ax.set_xlabel("$\\zeta$")
        ax.set_ylabel("$f_n(\\zeta)$")
        ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
        ax.set_xticks([0, np.pi / basis.NFP, 2 * np.pi / basis.NFP])
        ax.set_xticklabels(["$0$", "$\\pi/NFP$", "$2\\pi/NFP$"])
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax.set_title(
            "{}, $N={}$, $NFP={}$".format(basis.__class__.__name__, basis.N, basis.NFP),
        )
        fig.set_tight_layout(True)
        return fig, ax

    elif basis.__class__.__name__ == "DoubleFourierSeries":
        nmax = abs(basis.modes[:, 2]).max()
        mmax = abs(basis.modes[:, 1]).max()
        grid = LinearGrid(1, 100, 100, NFP=basis.NFP, endpoint=True)
        t = grid.nodes[:, 1].reshape((100, 100))
        z = grid.nodes[:, 2].reshape((100, 100))
        fig = plt.figure(
            # 2 * mmax + 1,
            # 2 * nmax + 1,
            figsize=kwargs.get("figsize", (nmax * 4 + 1, mmax * 4 + 1)),
            # sharex=True,
            # sharey=True,
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
                fi.reshape((100, 100)),
                levels=100,
                vmin=-1,
                vmax=1,
                cmap=kwargs.get("cmap", "coolwarm"),
            )
            if m == mmax:
                ax[mmax + m, nmax + n].set_xlabel(
                    "$\\zeta$ \n $n={}$".format(n), fontsize=10
                )
                ax[mmax + m, nmax + n].set_xticklabels(
                    ["$0$", None, "$\\pi/NFP$", None, "$2\\pi/NFP$"], fontsize=8
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
            "{}, $M={}$, $N={}$, $NFP={}$".format(
                basis.__class__.__name__, basis.M, basis.N, basis.NFP
            ),
            y=0.98,
        )
        return fig, ax

    elif basis.__class__.__name__ in ["ZernikePolynomial", "FourierZernikeBasis"]:
        lmax = abs(basis.modes[:, 0]).max().astype(int)
        mmax = abs(basis.modes[:, 1]).max().astype(int)

        grid = LinearGrid(100, 100, 1, endpoint=True)
        r = np.unique(grid.nodes[:, 0])
        v = np.unique(grid.nodes[:, 1])

        fig = plt.figure(figsize=kwargs.get("figsize", (3 * mmax, 3 * lmax / 2)))

        ax = {i: {} for i in range(lmax + 1)}
        ratios = np.ones(2 * (mmax + 1) + 1)
        ratios[-1] = kwargs.get("cbar_ratio", 0.25)
        gs = matplotlib.gridspec.GridSpec(
            lmax + 2, 2 * (mmax + 1) + 1, width_ratios=ratios
        )

        modes = basis.modes[np.where(basis.modes[:, 2] == 0)]
        Zs = basis.evaluate(grid.nodes, modes=modes)
        for i, (l, m) in enumerate(
            zip(modes[:, 0].astype(int), modes[:, 1].astype(int))
        ):
            Z = Zs[:, i].reshape((100, 100))
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

        cb_ax = plt.subplot(gs[:, -1])
        plt.subplots_adjust(right=0.8)
        cbar = fig.colorbar(im, cax=cb_ax)
        cbar.set_ticks(np.linspace(-1, 1, 9))
        fig.suptitle(
            "{}, $L={}$, $M={}$, spectral indexing = {}".format(
                basis.__class__.__name__, basis.L, basis.M, basis.spectral_indexing
            ),
            y=0.98,
        )
        fig.set_tight_layout(True)
        return fig, ax


def plot_logo(savepath=None, **kwargs):
    """Plot the DESC logo.

    Parameters
    ----------
    savepath : str or path-like
        path to save the figure to.
        File format is inferred from the filename (Default value = None)
    **kwargs :
        additional plot formatting parameters.
        options include ``'Dcolor'``, ``'Dcolor_rho'``, ``'Dcolor_theta'``,
        ``'Ecolor'``, ``'Scolor'``, ``'Ccolor'``, ``'BGcolor'``, ``'fig_width'``

    Returns
    -------
    fig : matplotlib.figure.Figure
        handle to the figure used for plotting
    ax : matplotlib.axes.Axes
        handle to the axis used for plotting

    Examples
    --------

    .. image:: ../../_static/images/plotting/plot_logo.png

    .. code-block:: python

        from desc.plotting import plot_logo
        plot_logo(savepath='../_static/images/plotting/plot_logo.png')

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
    Dcolor = kwargs.get("Dcolor", "xkcd:neon purple")
    Dcolor_rho = kwargs.get("Dcolor_rho", "xkcd:neon pink")
    Dcolor_theta = kwargs.get("Dcolor_theta", "xkcd:neon pink")
    Ecolor = kwargs.get("Ecolor", "deepskyblue")
    Scolor = kwargs.get("Scolor", "deepskyblue")
    Ccolor = kwargs.get("Ccolor", "deepskyblue")
    BGcolor = kwargs.get("BGcolor", "clear")
    fig_width = kwargs.get("fig_width", 3)
    fig_height = fig_width / 2
    contour_lw_ratio = kwargs.get("contour_lw_ratio", 0.3)
    lw = fig_width ** 0.5

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
        color=Dcolor_rho,
        lw=lw * contour_lw_ratio,
        ls="-",
    )
    # plot theta contours
    ax.plot(
        R[:, ::tstep],
        Z[:, ::tstep],
        color=Dcolor_theta,
        lw=lw * contour_lw_ratio,
        ls="-",
    )
    ax.plot(bdryR, bdryZ, color=Dcolor, lw=lw)

    if onlyD:
        if savepath is not None:
            fig.savefig(savepath, facecolor=fig.get_facecolor(), edgecolor="none")

        return fig, ax

    # E
    ax.plot([Eleft, Eleft + 1], [bottom, top], lw=lw, color=Ecolor, linestyle="-")
    ax.plot([Eleft, Eright], [bottom, bottom], lw=lw, color=Ecolor, linestyle="-")
    ax.plot(
        [Eleft + 1 / 2, Eright],
        [bottom + (top + bottom) / 2, bottom + (top + bottom) / 2],
        lw=lw,
        color=Ecolor,
        linestyle="-",
    )
    ax.plot([Eleft + 1, Eright], [top, top], lw=lw, color=Ecolor, linestyle="-")

    # S
    Sy = np.linspace(bottom, top + Soffset, 1000)
    Sx = Sw * np.cos(Sy * 3 / 2 * np.pi / (Sy.max() - Sy.min()) - np.pi) ** 2 + Sleft
    ax.plot(Sx, Sy[::-1] - Soffset / 2, lw=lw, color=Scolor, linestyle="-")

    # C
    Cx = Cw / 2 * np.cos(Ctheta) + Cx0
    Cy = Ch / 2 * np.sin(Ctheta) + Cy0
    ax.plot(Cx, Cy, lw=lw, color=Ccolor, linestyle="-")

    if savepath is not None:
        fig.savefig(savepath, facecolor=fig.get_facecolor(), edgecolor="none")

    return fig, ax


def plot_field_lines_sfl(eq, rho, seed_thetas=0, phi_end=2 * np.pi, ax=None, **kwargs):
    """Plots field lines on specified flux surface.

    Traces field lines at specified initial vartheta (:math:`\\vartheta`) seed
    locations, then plots them.
    Field lines traced by first finding the corresponding straight-field-line (SFL)
    coordinates :math:`(\\rho,\\vartheta,\\phi)` for each field line, then
    converting those to the computational :math:`(\\rho,\\theta,\\phi)` coordinates,
    then finally computing from those the toroidal :math:`(R,\\phi,Z)` coordinates of
    each field line.

    The SFL angle coordinates are found with the SFL relation:

        :math:`\\vartheta = \\iota \\phi + \\vartheta_0`

    Parameters
    ----------
    eq : Equilibrium
        Object from which to plot.
    rho : float
        Flux surface to trace field lines at.
    seed_thetas : float or array-like of floats
        Poloidal positions at which to seed magnetic field lines.
        If array-like, will plot multiple field lines.
    phi_end: float
        Toroidal angle to integrate field line until, in radians. Default is 2*pi.
    ax : matplotlib AxesSubplot, optional
        Axis to plot on.


    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure being plotted to.
    ax : matplotlib.axes.Axes or ndarray of Axes
        Axes being plotted to.
    field_line_coords : dict
        Dict containing the R,phi,Z coordinates of each field line traced.
        Dictionary entries are lists corresponding to the field lines for
        each seed_theta given. Also contains the scipy IVP solutions for info
        on how each line was integrated

    """
    if rho == 0:
        raise NotImplementedError(
            "Currently does not support field line tracing of the magnetic axis, please input 0 < rho < 1"
        )

    fig, ax = _format_ax(ax, is3d=True, figsize=kwargs.get("figsize", None))

    # check how many field lines to plot
    if seed_thetas is list:
        n_lines = len(seed_thetas)
    elif isinstance(seed_thetas, np.ndarray):
        n_lines = seed_thetas.size
    else:
        n_lines = 1

    phi0 = kwargs.get("phi0", 0)
    dphi = kwargs.get("dphi", 1e-2)  # spacing between points in phi, in radians
    N_pts = int((phi_end - phi0) / dphi)

    grid_single_rho = Grid(
        nodes=np.array([[rho, 0, 0]])
    )  # grid to get the iota value at the specified rho surface
    iota = eq.compute("iota", grid_single_rho)["iota"][0]

    varthetas = []
    phi = np.linspace(phi0, phi_end, N_pts)
    if n_lines > 1:
        for i in range(n_lines):
            varthetas.append(
                seed_thetas[i] + iota * phi
            )  # list of varthetas corresponding to the field line
    else:
        varthetas.append(
            seed_thetas + iota * phi
        )  # list of varthetas corresponding to the field line
    theta_coords = (
        []
    )  # list of nodes in (rho,theta,phi) corresponding to each (rho,vartheta,phi) node list
    print(
        "Calculating field line (rho,theta,zeta) coordinates corresponding to sfl coordinates"
    )
    for vartheta_list in varthetas:
        rhos = rho * np.ones_like(vartheta_list)
        sfl_coords = np.vstack((rhos, vartheta_list, phi)).T
        theta_coords.append(eq.compute_theta_coords(sfl_coords))

    # calculate R,phi,Z of nodes in grid
    # only need to do this after finding the grid corresponding to desired rho, vartheta, phi
    print(
        "Calculating field line (R,phi,Z) coordinates corresponding to (rho,theta,zeta) coordinates"
    )
    field_line_coords = {"Rs": [], "Zs": [], "phis": [], "seed_thetas": seed_thetas}
    for coords in theta_coords:
        grid = Grid(nodes=coords)
        toroidal_coords = eq.compute("R", grid)
        field_line_coords["Rs"].append(toroidal_coords["R"])
        field_line_coords["Zs"].append(toroidal_coords["Z"])
        field_line_coords["phis"].append(phi)

    for i in range(n_lines):
        xline = np.asarray(field_line_coords["Rs"][i]) * np.cos(
            field_line_coords["phis"][i]
        )
        yline = np.asarray(field_line_coords["Rs"][i]) * np.sin(
            field_line_coords["phis"][i]
        )

        ax.plot(xline, yline, field_line_coords["Zs"][i], linewidth=2)

    ax.set_xlabel(_axis_labels_XYZ[0])
    ax.set_ylabel(_axis_labels_XYZ[1])
    ax.set_zlabel(_axis_labels_XYZ[2])
    ax.set_title(
        "%d Magnetic Field Lines Traced On $\\rho=%1.2f$ Surface" % (n_lines, rho)
    )
    fig.set_tight_layout(True)

    # need this stuff to make all the axes equal, ax.axis('equal') doesnt work for 3d
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    return fig, ax, field_line_coords


def plot_field_lines_real_space(
    eq,
    rho,
    seed_thetas=0,
    phi_end=2 * np.pi,
    grid=None,
    ax=None,
    B_interp=None,
    return_B_interp=False,
    **kwargs,
):
    """***Use plot_field_lines_sfl if plotting from a solved equilibrium, as that is faster and more accurate than real space interpolation***
    Traces field lines on specified flux surface at specified initial theta seed locations, then plots them.
    Field lines integrated by first fitting the magnetic field with radial basis functions (RBF) in R,Z,phi, then integrating the field line
    from phi=0 up to the specified phi angle, by solving:

    :math:`\\frac{dR}{d\\phi} = \\frac{RB_R}{B_{\\phi}} , \\frac{dZ}{d\\phi} = \\frac{RB_Z}{B_{\\phi}}`

    :math:`B_R = \\mathbf{B} \\cdot \\hat{\\mathbf{R}} = (B^{\\theta} \\mathbf{e}_{\\theta} + B^{\\zeta} \\mathbf{e}_{\\zeta}) \\cdot \\hat{\\mathbf{R}} = B^{\\theta} \\frac{\\partial R}{\\partial \\theta} + B^{\\zeta} \\frac{\\partial R}{\\partial \\zeta}`

    :math:`B_Z = \\mathbf{B} \\cdot \\hat{\\mathbf{Z}} = (B^{\\theta} \\mathbf{e}_{\\theta} + B^{\\zeta} \\mathbf{e}_{\\zeta}) \\cdot \\hat{\\mathbf{Z}} = B^{\\theta} \\frac{\\partial Z}{\\partial \\theta} + B^{\\zeta} \\frac{\\partial Z}{\\partial \\zeta}`

    :math:`B_{\\phi} = \\mathbf{B} \\cdot \\hat{\\mathbf{\\phi}} = R B^{\\zeta}`

    Parameters
    ----------
    eq : Equilibrium
        object from which to plot
    rho : float
        flux surface to trace field lines at
    seed_thetas : float or array-like of floats
        theta positions at which to seed magnetic field lines, if array-like, will plot multiple field lines
    phi_end: float
        phi to integrate field line until, in radians. Default is 2*pi
    grid : Grid, optional
        grid of rho, theta, zeta coordinates used to evaluate magnetic field at, which is then interpolated with RBF
    ax : matplotlib AxesSubplot, optional
        axis to plot on
    B_interp : dict of scipy.interpolate.rbf.Rbf or equivalent call signature interplators, optional
        if not None, uses the passed-in interpolation objects instead of fitting the magnetic field with Rbf's. Useful
        if have already ran plot_field_lines once and want to change the seed thetas or how far to integrate in phi.
        Dict should have the following keys: ['B_R'], ['B_Z'], and ['B_phi'], corresponding to the interplating object for
        each cylindrical component of the magnetic field.
    return_B_interp: bool, default False
        If true, in addition to returning the fig, axis and field line coordinates, will also return the dictionary of interpolating radial basis functions
        interpolating the magnetic field in (R,phi,Z)


    Returns
    -------
    fig : matplotlib.figure.Figure
        figure being plotted to
    ax : matplotlib.axes.Axes or ndarray of Axes
        axes being plotted to
    field_line_coords : dict
        dict containing the R,phi,Z coordinates of each field line traced. Dictionary entries are lists
        corresponding to the field lines for each seed_theta given. Also contains the scipy IVP solutions for info
        on how each line was integarted
    B_interp : dict, only returned if return_B_interp is True
        dict of scipy.interpolate.rbf.Rbf or equivalent call signature interplators, which interpolate the cylindrical
        components of magnetic field in (R,phi,Z)
        Dict has the following keys: ['B_R'], ['B_Z'], and ['B_phi'], corresponding to the interplating object for
        each cylindrical component of the magnetic field, and the interpolators have call signature
        B(R,phi,Z) = interpolator(R,phi,Z)

    """
    nfp = 1
    if grid is None:
        grid_kwargs = {"M": 30, "N": 30, "L": 20, "NFP": nfp, "axis": False}
        grid = _get_grid(**grid_kwargs)

    fig, ax = _format_ax(ax, is3d=True, figsize=kwargs.get("figsize", None))

    # check how many field lines to plot
    if seed_thetas is list:
        n_lines = len(seed_thetas)
    elif isinstance(seed_thetas, np.ndarray):
        n_lines = seed_thetas.size
    else:
        n_lines = 1
    phi0 = kwargs.get("phi0", 0)

    # calculate toroidal coordinates
    toroidal_coords = eq.compute("phi", grid)
    Rs = toroidal_coords["R"]
    Zs = toroidal_coords["Z"]
    phis = toroidal_coords["phi"]

    # calculate cylindrical B
    magnetic_field = eq.compute("B", grid)
    BR = magnetic_field["B_R"]
    BZ = magnetic_field["B_Z"]
    Bphi = magnetic_field["B_phi"]

    if B_interp is None:  # must fit RBfs to interpolate B field in R,phi,Z
        print(
            "Fitting magnetic field with radial basis functions in R,phi,Z (may take a few minutes)"
        )
        BRi = Rbf(Rs, Zs, phis, BR)
        BZi = Rbf(Rs, Zs, phis, BZ)
        Bphii = Rbf(Rs, Zs, phis, Bphi)
        B_interp = {"B_R": BRi, "B_Z": BZi, "B_phi": Bphii}

    field_line_coords = {
        "Rs": [],
        "Zs": [],
        "phis": [],
        "IVP solutions": [],
        "seed_thetas": seed_thetas,
    }
    if n_lines > 1:
        for theta in seed_thetas:
            field_line_Rs, field_line_phis, field_line_Zs, sol = _field_line_Rbf(
                rho, theta, phi_end, grid, Rs, Zs, B_interp, phi0
            )
            field_line_coords["Rs"].append(field_line_Rs)
            field_line_coords["Zs"].append(field_line_Zs)
            field_line_coords["phis"].append(field_line_phis)
            field_line_coords["IVP solutions"].append(sol)

    else:
        field_line_Rs, field_line_phis, field_line_Zs, sol = _field_line_Rbf(
            rho, seed_thetas, phi_end, grid, Rs, Zs, B_interp, phi0
        )
        field_line_coords["Rs"].append(field_line_Rs)
        field_line_coords["Zs"].append(field_line_Zs)
        field_line_coords["phis"].append(field_line_phis)
        field_line_coords["IVP solutions"].append(sol)

    for i, solution in enumerate(field_line_coords["IVP solutions"]):
        if not solution.success:
            print(
                "Integration from seed theta %1.2f radians was not successful!"
                % seed_thetas[i]
            )

    for i in range(n_lines):
        xline = np.asarray(field_line_coords["Rs"][i]) * np.cos(
            field_line_coords["phis"][i]
        )
        yline = np.asarray(field_line_coords["Rs"][i]) * np.sin(
            field_line_coords["phis"][i]
        )

        ax.plot(xline, yline, field_line_coords["Zs"][i], linewidth=2)

    ax.set_xlabel(_axis_labels_XYZ[0])
    ax.set_ylabel(_axis_labels_XYZ[1])
    ax.set_zlabel(_axis_labels_XYZ[2])
    ax.set_title(
        "%d Magnetic Field Lines Traced On $\\rho=%1.2f$ Surface" % (n_lines, rho)
    )
    fig.set_tight_layout(True)

    # need this stuff to make all the axes equal, ax.axis('equal') doesnt work for 3d
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    if return_B_interp:
        return fig, ax, field_line_coords, B_interp
    else:
        return fig, ax, field_line_coords


def _find_idx(rho0, theta0, phi0, grid):
    """
    Finds the node index corresponding to the rho,theta,zeta node closest to the given rho0,theta0,phi0

    Parameters
    ----------
    rho0 : float
        rho to find closest grid point to.
    theta0 : float
        theta to find closest grid point to.
    phi0 : float
        phi to find closest grid point to.
    grid : Grid
        grid to find closest point on


    Returns
    -------
    idx_pt : int
        index of the grid node closest to the given point.

    """
    rhos = grid.nodes[:, 0]
    thetas = grid.nodes[:, 1]
    phis = grid.nodes[:, 2]

    if theta0 < 0:
        theta0 = 2 * np.pi + theta0
    if theta0 > 2 * np.pi:
        theta0 == np.mod(theta0, 2 * np.pi)
    if phi0 < 0:
        phi0 = 2 * np.pi + phi0
    if phi0 > 2 * np.pi:
        phi0 == np.mod(phi0, 2 * np.pi)

    bool1 = np.logical_and(
        np.abs(rhos - rho0) == np.min(np.abs(rhos - rho0)),
        np.abs(thetas - theta0) == np.min(np.abs(thetas - theta0)),
    )
    bool2 = np.logical_and(bool1, np.abs(phis - phi0) == np.min(np.abs(phis - phi0)))
    idx_pt = np.where(bool2 == True)[0][0]
    return idx_pt


def _field_line_Rbf(rho, theta0, phi_end, grid, Rs, Zs, B_interp, phi0=0):
    """Takes the initial poloidal angle you want to seed a field line at (at phi=0),
    and integrates along the field line to the specified phi_end. returns fR,fZ,fPhi,
    the R,Z,Phi coordinates of the field line trajectory."""

    fR = []
    fZ = []
    fPhi = []
    idx0 = _find_idx(rho, theta0, phi0, grid)
    curr_R = Rs[idx0]
    curr_Z = Zs[idx0]
    fR.append(curr_R)
    fZ.append(curr_Z)
    fPhi.append(phi0)

    # integrate field lines in Phi
    print(
        "Integrating Magnetic Field Line Equation from seed theta = %f radians" % theta0
    )
    y0 = [fR[0], fZ[0]]

    def rhs(phi, y):
        """RHS of magnetic field line equation."""
        dRdphi = (
            y[0]
            * B_interp["B_R"](y[0], y[1], np.mod(phi, 2 * np.pi))
            / B_interp["B_phi"](y[0], y[1], np.mod(phi, 2 * np.pi))
        )
        dZdphi = (
            y[0]
            * B_interp["B_Z"](y[0], y[1], np.mod(phi, 2 * np.pi))
            / B_interp["B_phi"](y[0], y[1], np.mod(phi, 2 * np.pi))
        )
        return [dRdphi, dZdphi]

    n_tries = 1
    max_step = 0.01
    sol = solve_ivp(rhs, [0, phi_end], y0, max_step=max_step)
    while not sol.success and n_tries < 4:
        max_step = 0.5 * max_step
        n_tries += 1
        sol = solve_ivp(rhs, [0, phi_end], y0, max_step=max_step)
    fR = sol.y[0, :]
    fZ = sol.y[1, :]
    fPhi = sol.t
    return fR, fPhi, fZ, sol
