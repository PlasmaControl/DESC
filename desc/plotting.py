import os
from matplotlib import rcParams, cycler
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import re
from termcolor import colored

from desc.equilibrium_io import read_desc
from desc.grid import Grid, LinearGrid
from desc.transform import Transform
from desc.configuration import Configuration

colorblind_colors = [(0.0000, 0.4500, 0.7000),  # blue
                     (0.8359, 0.3682, 0.0000),  # vermillion
                     (0.0000, 0.6000, 0.5000),  # bluish green
                     (0.9500, 0.9000, 0.2500),  # yellow
                     (0.3500, 0.7000, 0.9000),  # sky blue
                     (0.8000, 0.6000, 0.7000),  # reddish purple
                     (0.9000, 0.6000, 0.0000)]  # orange
dashes = [(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # solid
          (3.7, 1.6, 0.0, 0.0, 0.0, 0.0),  # dashed
          (1.0, 1.6, 0.0, 0.0, 0.0, 0.0),  # dotted
          (6.4, 1.6, 1.0, 1.6, 0.0, 0.0),  # dot dash
          (3.0, 1.6, 1.0, 1.6, 1.0, 1.6),  # dot dot dash
          (6.0, 4.0, 0.0, 0.0, 0.0, 0.0),  # long dash
          (1.0, 1.6, 3.0, 1.6, 3.0, 1.6)]  # dash dash dot
matplotlib.rcdefaults()
rcParams['font.family'] = 'DejaVu Serif'
rcParams['mathtext.fontset'] = 'cm'
rcParams['font.size'] = 10
rcParams['figure.facecolor'] = (1, 1, 1, 1)
rcParams['figure.figsize'] = (6, 4)
rcParams['figure.dpi'] = 141
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
rcParams['axes.labelsize'] = 'small'
rcParams['axes.titlesize'] = 'medium'
rcParams['lines.linewidth'] = 1
rcParams['lines.solid_capstyle'] = 'round'
rcParams['lines.dash_capstyle'] = 'round'
rcParams['lines.dash_joinstyle'] = 'round'
rcParams['xtick.labelsize'] = 'x-small'
rcParams['ytick.labelsize'] = 'x-small'
rcParams['text.usetex'] = True
color_cycle = cycler(color=colorblind_colors)
dash_cycle = cycler(dashes=dashes)
rcParams['axes.prop_cycle'] = color_cycle


class Plot:
    """Class for plotting instances of Configuration and Equilibria on a linear grid.
    """
    axis_labels_rtz = [r'$\rho$', r'$\theta$', r'$\zeta$']
    axis_labels_RPZ = [r'$R$', r'$\phi$', r'$Z$']
    axis_labels_XYZ = [r'$X$', r'$Y$', r'$Z$']

    def __init__(self):
        """Initialize a Plot class.

        Parameters
        ----------

        Returns
        -------
        None

        """
        pass

    def format_ax(self, ax, is3d=False):
        """Check type of ax argument. If ax is not a matplotlib AxesSubplot, initalize one.

        Parameters
        ----------
        ax : None or matplotlib AxesSubplot instance
            DESCRIPTION
        is3d: bool
            default is False

        Returns
        -------
        matpliblib Figure instance, matplotlib AxesSubplot instance

        """
        if ax is None:
            if is3d:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                return fig, ax
            else:
                fig, ax = plt.subplots()
                return fig, ax
# FIXME: cannot check types against matplotlib.axes._subplots.AxesSubplot,
# as it throws an error that it has no such attribute
        elif type(ax) is matplotlib.axes._subplots.AxesSubplot or matplotlib.axes._subplots.Axes3DSubplot:
            return plt.gcf(), ax
        else:
            raise TypeError(
                colored("ax agument must be None or an axis instance", 'red'))

    def get_grid(self, **kwargs):
        """Get grid for plotting.

        Parameters
        ----------
        kwargs
            any arguments taken by LinearGrid (Default L=100, M=1, N=1)

        Returns
        -------
        LinearGrid

        """
        grid_args = {'L': 1, 'M': 1, 'N': 1, 'NFP': 1, 'sym': False,
                     'endpoint': True, 'rho': None, 'theta': None, 'zeta': None}
        for key in kwargs.keys():
            if key in grid_args.keys():
                grid_args[key] = kwargs[key]
        grid = LinearGrid(**grid_args)

        plot_axes = [0, 1, 2]
        if grid.L == 1:
            plot_axes.remove(0)
        if grid.M == 1:
            plot_axes.remove(1)
        if grid.N == 1:
            plot_axes.remove(2)

        return grid, tuple(plot_axes)

    def plot_1d(self, eq: Configuration, name: str, grid: Grid = None, ax=None, **kwargs):
        """Plots 1D profiles.

        Parameters
        ----------
        eq : Configuration
            object from which to plot
        name : str
            name of variable to plot
        grid : Grid, optional
            grid of coordinates to plot at
        ax : matplotlib AxesSubplot, optional
            axis to plot on
        kwargs
            any arguments taken by LinearGrid

        Returns
        -------
        axis

        """
        if grid is None:
            if kwargs == {}:
                kwargs.update({'L': 100, 'NFP': eq.NFP})
            grid, plot_axes = self.get_grid(**kwargs)
        if len(plot_axes) != 1:
            return ValueError(colored("Grid must be 1D", 'red'))

        name_dict = self.format_name(name)
        data = self.compute(eq, name_dict, grid)
        fig, ax = self.format_ax(ax)

        # reshape data to 1D
        data = data[:, 0, 0]

        ax.plot(grid.nodes[:, plot_axes[0]], data)

        ax.set_xlabel(self.axis_labels_rtz[plot_axes[0]])
        ax.set_ylabel(self.name_label(name_dict))
        return ax

    def plot_2d(self, eq: Configuration, name: str, grid: Grid = None, ax=None, **kwargs):
        """Plots 2D cross-sections.

        Parameters
        ----------
        eq : Configuration
            object from which to plot
        name : str
            name of variable to plot
        grid : Grid, optional
            grid of coordinates to plot at
        ax : matplotlib AxesSubplot, optional
            axis to plot on
        kwargs
            any arguments taken by LinearGrid

        Returns
        -------
        axis

        """
        if grid is None:
            if kwargs == {}:
                kwargs.update({'L': 25, 'M': 25, 'NFP': eq.NFP})
            grid, plot_axes = self.get_grid(**kwargs)
        if len(plot_axes) != 2:
            return ValueError(colored("Grid must be 2D", 'red'))

        name_dict = self.format_name(name)
        data = self.compute(eq, name_dict, grid)
        fig, ax = self.format_ax(ax)
        divider = make_axes_locatable(ax)

        # reshape data to 2D
        if 0 in plot_axes:
            if 1 in plot_axes:      # rho & theta
                data = data[:, :, 0]
            else:                   # rho & zeta
                data = data[:, 0, :]
        else:                       # theta & zeta
            data = data[0, :, :]

        imshow_kwargs = {'origin': 'lower',
                         'interpolation': 'bilinear',
                         'aspect': 'auto'}
        imshow_kwargs['extent'] = [grid.nodes[0, plot_axes[0]],
                                   grid.nodes[-1, plot_axes[0]
                                              ], grid.nodes[0, plot_axes[1]],
                                   grid.nodes[-1, plot_axes[1]]]
        cax_kwargs = {'size': '5%',
                      'pad': 0.05}

        im = ax.imshow(data, **imshow_kwargs)
        cax = divider.append_axes('right', **cax_kwargs)
        cbar = fig.colorbar(im, cax=cax)
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()

        ax.set_xlabel(self.axis_labels_rtz[plot_axes[1]])
        ax.set_ylabel(self.axis_labels_rtz[plot_axes[0]])
        ax.set_title(self.name_label(name_dict))
        return ax

    def plot_3d(self, eq: Configuration, name: str, grid: Grid = None, ax=None, **kwargs):
        """Plots 3D surfaces.

        Parameters
        ----------
        eq : Configuration
            object from which to plot
        name : str
            name of variable to plot
        grid : Grid, optional
            grid of coordinates to plot at
        ax : matplotlib AxesSubplot, optional
            axis to plot on
        kwargs
            any arguments taken by LinearGrid

        Returns
        -------
        axis

        """
        if grid is None:
            if kwargs == {}:
                kwargs.update({'M': 46, 'N': 46, 'NFP': eq.NFP})
            grid, plot_axes = self.get_grid(**kwargs)
        if len(plot_axes) != 2:
            return ValueError(colored("Grid must be 2D", 'red'))

        name_dict = self.format_name(name)
        data = self.compute(eq, name_dict, grid)
        fig, ax = self.format_ax(ax, is3d=True)

        coords = eq.compute_cartesian_coords(grid)
        X = coords['X'].reshape((grid.L, grid.M, grid.N), order='F')
        Y = coords['Y'].reshape((grid.L, grid.M, grid.N), order='F')
        Z = coords['Z'].reshape((grid.L, grid.M, grid.N), order='F')

        # reshape data to 2D
        if 0 in plot_axes:
            if 1 in plot_axes:      # rho & theta
                data = data[:, :, 0]
                X = X[:, :, 0]
                Y = Y[:, :, 0]
                Z = Z[:, :, 0]
            else:                   # rho & zeta
                data = data[:, 0, :]
                X = X[:, 0, :]
                Y = Y[:, 0, :]
                Z = Z[:, 0, :]
        else:                       # theta & zeta
            data = data[0, :, :]
            X = X[0, :, :]
            Y = Y[0, :, :]
            Z = Z[0, :, :]

        minn, maxx = data.min(), data.max()
        m = plt.cm.ScalarMappable()
        m.set_array([])
        fcolors = m.to_rgba(data)

        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=fcolors,
                        vmin=minn, vmax=maxx)
        fig.colorbar(m)

        ax.set_xlabel(self.axis_labels_XYZ[0])
        ax.set_ylabel(self.axis_labels_XYZ[1])
        ax.set_zlabel(self.axis_labels_XYZ[2])
        ax.set_title(self.name_label(name_dict))
        return ax

    def plot_surfaces(self, eq: Configuration, grid: Grid = None, ax=None, **kwargs):
        """Plots flux surfaces.

        Parameters
        ----------
        eq : Configuration
            object from which to plot
        name : str
            name of variable to plot
        grid : Grid, optional
            grid of coordinates to plot at
        ax : matplotlib AxesSubplot, optional
            axis to plot on
        kwargs
            any arguments taken by LinearGrid

        Returns
        -------
        axis

        """
        if grid is None:
            if kwargs == {}:
                kwargs.update({'L': 6, 'M': 180})
            grid, plot_axes = self.get_grid(**kwargs)
        if len(plot_axes) != 2:
            return ValueError(colored("Grid must be 2D", 'red'))
        if 2 in plot_axes:
            return ValueError(colored("Grid must be in rho vs theta", 'red'))

        coords = eq.compute_toroidal_coords(grid)
        R = coords['R'].reshape((grid.L, grid.M, grid.N), order='F')[:, :, 0]
        Z = coords['Z'].reshape((grid.L, grid.M, grid.N), order='F')[:, :, 0]

        fig, ax = self.format_ax(ax)

        ax.plot(R[0, 0], Z[0, 0], 'bo')
        ax.plot(R.T, Z.T, 'b-')

        ax.axis('equal')
        ax.set_xlabel(self.axis_labels_RPZ[0])
        ax.set_ylabel(self.axis_labels_RPZ[2])

        return ax

    def compute(self, eq: Configuration, name: str, grid: Grid):
        """Compute value specified by name on grid for equilibrium eq.

        Parameters
        ----------
        eq : Configuration
            object from which to plot
        name : str
            name of variable to plot
        grid : Grid, optional
            grid of coordinates to plot at

        Returns
        -------
        out, float array of shape (L, M, N)
            computed values

        """
        if type(name) is not dict:
            name_dict = self.format_name(name)
        else:
            name_dict = name

        # primary calculations
        if name_dict['base'] in ['r','lambda']:
            out = eq.compute_polar_coords(grid)[self.__name_key__(name_dict)]
        elif name_dict['base'] in ['R', 'Z']:
            out = eq.compute_toroidal_coords(grid)[self.__name_key__(name_dict)]
        elif name_dict['base'] in ['p', 'iota', 'psi', 'chi']:
            out = eq.compute_profiles(grid)[self.__name_key__(name_dict)]
        elif name_dict['base'] == 'g':
            out = eq.compute_jacobian(grid)[self.__name_key__(name_dict)]
        elif name_dict['base'] == 'B':
            out = eq.compute_magnetic_field(grid)[self.__name_key__(name_dict)]
        elif name_dict['base'] == 'J':
            out = eq.compute_current_density(
                grid)[self.__name_key__(name_dict)]
        elif name_dict['base'] == '|B|':
            out = eq.compute_magnetic_field_magnitude(
                grid)[self.__name_key__(name_dict)]
        elif name_dict['base'] == '|F|':
            out = eq.compute_force_error_magnitude(
                grid)[self.__name_key__(name_dict)]
        else:
            raise NotImplementedError(
                "No output for base named '{}'.".format(name_dict['base']))

        # secondary calculations
        power = name_dict['power']
        if power != '':
            try:
                power = float(power)
            except ValueError:
                # handle fractional exponents
                if '/' in power:
                    frac = power.split('/')
                    power = frac[0] / frac[1]
                else:
                    raise ValueError(
                        "Could not convert string to float: '{}'".format(power))
            out = out**power

        return out.reshape((grid.L, grid.M, grid.N), order='F')

    def format_name(self, name):
        """Parse name string into dictionary.

        Parameters
        ----------
        name : str

        Returns
        -------
        parsed name : dict

        """
        name_dict = {'base': '', 'sups': '', 'subs': '', 'power': '', 'd': ''}
        if '**' in name:
            parsename, power = name.split('**')
            if '_' in power or '^' in power:
                raise SyntaxError(
                    'Power operands must come after components and derivatives.')
        else:
            power = ''
            parsename = name
        name_dict['power'] += power
        if '_' in parsename:
            split = parsename.split('_')
            if len(split) == 3:
                name_dict['base'] += split[0]
                name_dict['subs'] += split[1]
                name_dict['d'] += split[2]
            elif '^' in split[0]:
                name_dict['base'], name_dict['sups'] = split[0].split('^')
                name_dict['d'] = split[1]
            elif len(split) == 2:
                name_dict['base'], other = split
                if other in ['rho', 'theta', 'zeta']:
                    name_dict['subs'] = other
                else:
                    name_dict['d'] = other
            else:
                raise SyntaxError('String format is not valid.')
        elif '^' in parsename:
            name_dict['base'], name_dict['sups'] = parsename.split('^')
        else:
            name_dict['base'] = parsename
        return name_dict

    def name_label(self, name_dict):
        """Create label for name dictionary.

        Parameters
        ----------
        name_dict : dict
            name dictionary created by format_name method

        Returns
        -------
        label : str

        """
        esc = r'\\'[:-1]

        if 'mag' in name_dict['base']:
            base = '|' + re.sub('mag', '', name_dict['base']) + '|'
        else:
            base = name_dict['base']

        if name_dict['d'] != '':
            dstr0 = 'd'
            dstr1 = '/d' + name_dict['d']
            if name_dict['power'] != '':
                dstr0 = '(' + dstr0
                dstr1 = dstr1 + ')^{' + name_dict['power'] + '}'
            else:
                pass
        else:
            dstr0 = ''
            dstr1 = ''

        if name_dict['power'] != '':
            if name_dict['d'] != '':
                pstr = ''
            else:
                pstr = name_dict['power']
        else:
            pstr = ''

        if name_dict['sups'] != '':
            supstr = '^{' + esc + name_dict['sups'] + ' ' + pstr + '}'
        elif pstr != '':
            supstr = '^{' + pstr + '}'
        else:
            supstr = ''

        if name_dict['subs'] != '':
            substr = '_{' + esc + name_dict['subs'] + '}'
        else:
            substr = ''
        label = r'$' + dstr0 + base + supstr + substr + dstr1 + '$'
        return label

    def __name_key__(self, name_dict):
        """Reconstruct name for dictionary key used in Configuration compute methods.

        Parameters
        ----------
        name_dict : dict
            name dictionary created by format_name method

        Returns
        -------
        name_key : str

        """
        out = name_dict['base']
        if name_dict['sups'] != '':
            out += '^' + name_dict['sups']
        if name_dict['subs'] != '':
            out += '_' + name_dict['subs']
        if name_dict['d'] != '':
            out += '_' + name_dict['d']
        return out


def plot_logo(savepath=None, **kwargs):
    """Plots the DESC logo

    Parameters
    ----------
    savepath : str or path-like
        path to save the figure to.
        File format is inferred from the filename (Default value = None)
    **kwargs :
        additional plot formatting parameters.
        options include 'Dcolor', 'Dcolor_rho', 'Dcolor_theta',
        'Ecolor', 'Scolor', 'Ccolor', 'BGcolor', 'fig_width'


    Returns
    -------
    fig : matplotlib.figure
        handle to the figure used for plotting
    ax : matplotlib.axes
        handle to the axis used for plotting

    """
    onlyD = kwargs.get('onlyD', False)
    Dcolor = kwargs.get('Dcolor', 'xkcd:neon purple')
    Dcolor_rho = kwargs.get('Dcolor_rho', 'xkcd:neon pink')
    Dcolor_theta = kwargs.get('Dcolor_theta', 'xkcd:neon pink')
    Ecolor = kwargs.get('Ecolor', 'deepskyblue')
    Scolor = kwargs.get('Scolor', 'deepskyblue')
    Ccolor = kwargs.get('Ccolor', 'deepskyblue')
    BGcolor = kwargs.get('BGcolor', 'clear')
    fig_width = kwargs.get('fig_width', 3)
    fig_height = fig_width/2
    contour_lw_ratio = kwargs.get('contour_lw_ratio', 0.3)
    lw = fig_width**.5

    transparent = False
    if BGcolor == 'dark':
        BGcolor = 'xkcd:charcoal grey'
    elif BGcolor == 'light':
        BGcolor = 'white'
    elif BGcolor == 'clear':
        BGcolor = 'white'
        transparent = True

    path = os.path.dirname(os.path.abspath(__file__))
    equil = read_desc(path + '/../examples/DESC/outputs/LOGO_m12x18_n0x0')

    if onlyD:
        fig_width = fig_width/2
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_axes([0.1, 0.1, .8, .8])
    ax.axis('equal')
    ax.axis('off')
    ax.set_facecolor(BGcolor)
    fig.set_facecolor(BGcolor)
    if transparent:
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)

    bottom = 0
    top = 10
    Dleft = 0
    Dw = 8
    Dh = top-bottom + 2
    DX = Dleft + Dw/2
    DY = (top-bottom)/2
    Dright = Dleft + Dw

    Eleft = Dright + 0.5
    Eright = Eleft + 4

    Soffset = 1
    Sleft = Eright + 0.5
    Sw = 5
    Sright = Sleft + Sw

    Ctheta = np.linspace(np.pi/4, 2*np.pi-np.pi/4, 1000)
    Cleft = Sright + 0.75
    Cw = 4
    Ch = 11
    Cx0 = Cleft + Cw/2
    Cy0 = (top-bottom)/2

    # D
    cR = equil['cR']
    cZ = equil['cZ']
    zern_idx = equil['zern_idx']
    NFP = equil['NFP']
    R0, Z0 = axis_posn(cR, cZ, zern_idx, NFP)

    nr = kwargs.get('nr', 5)
    nt = kwargs.get('nt', 8)
    Nr = 100
    Nt = 361
    rstep = Nr//nr
    tstep = Nt//nt
    zeta = 0
    r = np.linspace(0, 1, Nr)
    t = np.linspace(0, 2*np.pi, Nt)
    r, t = np.meshgrid(r, t, indexing='ij')
    r = r.flatten()
    t = t.flatten()
    z = zeta*np.ones_like(r)
    zernike_transform = ZernikeTransform([r, t, z], zern_idx, NFP)
    bdry_nodes = np.array(
        [np.ones(Nt), np.linspace(0, 2*np.pi, Nt), np.ones(Nt)])
    bdry_zernike_transform = ZernikeTransform(bdry_nodes, zern_idx, NFP)

    R = zernike_transform.transform(cR, 0, 0, 0).reshape((Nr, Nt))
    Z = zernike_transform.transform(cZ, 0, 0, 0).reshape((Nr, Nt))
    bdryR = bdry_zernike_transform.transform(cR, 0, 0, 0)
    bdryZ = bdry_zernike_transform.transform(cZ, 0, 0, 0)

    R = (R-R0)/(R.max()-R.min())*Dw + DX
    Z = (Z-Z0)/(Z.max()-Z.min())*Dh + DY
    bdryR = (bdryR-R0)/(bdryR.max()-bdryR.min())*Dw + DX
    bdryZ = (bdryZ-Z0)/(bdryZ.max()-bdryZ.min())*Dh + DY

    # plot r contours
    ax.plot(R.T[:, ::rstep], Z.T[:, ::rstep],
            color=Dcolor_rho, lw=lw*contour_lw_ratio, ls='-')
    # plot theta contours
    ax.plot(R[:, ::tstep], Z[:, ::tstep],
            color=Dcolor_theta, lw=lw*contour_lw_ratio, ls='-')
    ax.plot(bdryR, bdryZ, color=Dcolor, lw=lw)

    if onlyD:
        if savepath is not None:
            fig.savefig(savepath, facecolor=fig.get_facecolor(),
                        edgecolor='none')

        return fig, ax

    # E
    ax.plot([Eleft, Eleft+1], [bottom, top],
            lw=lw, color=Ecolor, linestyle='-')
    ax.plot([Eleft, Eright], [bottom, bottom],
            lw=lw, color=Ecolor, linestyle='-')
    ax.plot([Eleft+1/2, Eright], [bottom+(top+bottom)/2, bottom +
                                  (top+bottom)/2], lw=lw, color=Ecolor, linestyle='-')
    ax.plot([Eleft+1, Eright], [top, top], lw=lw, color=Ecolor, linestyle='-')

    # S
    Sy = np.linspace(bottom, top+Soffset, 1000)
    Sx = Sw*np.cos(Sy*3/2*np.pi/(Sy.max()-Sy.min())-np.pi)**2 + Sleft
    ax.plot(Sx, Sy[::-1]-Soffset/2, lw=lw, color=Scolor, linestyle='-')

    # C
    Cx = Cw/2*np.cos(Ctheta)+Cx0
    Cy = Ch/2*np.sin(Ctheta)+Cy0
    ax.plot(Cx, Cy, lw=lw, color=Ccolor, linestyle='-')

    if savepath is not None:
        fig.savefig(savepath, facecolor=fig.get_facecolor(), edgecolor='none')

    return fig, ax


def plot_zernike_basis(M, delta_lm, indexing, **kwargs):
    """Plots spectral basis of zernike basis functions

    Parameters
    ----------
    M : int
        maximum poloidal resolution
    delta_lm : int
        maximum difference between radial mode l and poloidal mode m
    indexing : str
        zernike indexing method. One of 'fringe', 'ansi', 'house', 'chevron'
    **kwargs :
        additional plot formatting arguments


    Returns
    -------
    fig : matplotlib.figure
        handle to figure
    ax : dict of matplotlib.axes
        nested dictionary, ax[l][m] is the handle to the
        axis for radial mode l, poloidal mode m

    """

    cmap = kwargs.get('cmap', 'coolwarm')
    scale = kwargs.get('scale', 1)
    npts = kwargs.get('npts', 100)
    levels = kwargs.get('levels', np.linspace(-1, 1, npts))

    ls, ms, ns = get_zern_basis_idx_dense(M, 0, delta_lm, indexing).T
    lmax = np.max(ls)
    mmax = np.max(ms)

    r = np.linspace(0, 1, npts)
    v = np.linspace(0, 2*np.pi, npts)
    rr, vv = np.meshgrid(r, v, indexing='ij')

    fig = plt.figure(figsize=(scale*mmax, scale*lmax/2))

    ax = {i: {} for i in range(lmax+1)}
    gs = matplotlib.gridspec.GridSpec(lmax+1, 2*(mmax+1))

    Zs = zern(rr.flatten(), vv.flatten(), ls, ms, 0, 0)

    for i, (l, m) in enumerate(zip(ls, ms)):
        Z = Zs[:, i].reshape((npts, npts))
        ax[l][m] = plt.subplot(gs[l, m+mmax:m+mmax+2], projection='polar')
        ax[l][m].set_title('$\mathcal{Z}_{' + str(l) + '}^{' + str(m) + '}$')
        ax[l][m].axis('off')
        im = ax[l][m].contourf(v, r, Z, levels=levels, cmap=cmap)

    cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
    plt.subplots_adjust(right=.8)
    cbar = fig.colorbar(im, cax=cb_ax)
    cbar.set_ticks(np.linspace(-1, 1, 9))

    return fig, ax
