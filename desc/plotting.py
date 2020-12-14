import os
from matplotlib import rcParams, cycler
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import re
from abc import ABC, abstractmethod
from desc.equilibrium_io import read_desc
from desc.vmec import vmec_interpolate
from desc.grid import LinearGrid
from desc.transform import Transform
from desc.configuration import compute_coordinate_derivatives, compute_covariant_basis
from desc.configuration import compute_contravariant_basis, compute_jacobian
from desc.configuration import compute_magnetic_field, compute_plasma_current, compute_force_magnitude


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
# rcParams['text.usetex']=True
color_cycle = cycler(color=colorblind_colors)
dash_cycle = cycler(dashes=dashes)
rcParams['axes.prop_cycle'] = color_cycle


class Plot:
    """Class for plotting instances of Configuration and Equilibria on a linear grid.
    """
    axis_labels = [r'$R$', r'$\theta$', r'$\zeta$']
    def __init__(self):#grid='std', **kwargs):
        """Initialize a Plot class.

        Parameters
        __________

        Returns
        _______
        None

        """
        pass

    def __format_rtz__(self, rtz):
        type_rtz = type(rtz)
        if type_rtz is np.ndarray:
            return rtz
        elif type_rtz is list:
            return np.array(rtz)
        elif type_rtz is float:
            return np.array([rtz])
        else:
            raise TypeError('rho, theta, and zeta must be a numpy array, list '
                'of floats, or float.')

    def format_ax(self, ax):
        """Check type of ax argument. If ax is not a matplotlib AxesSubplot, initalize one.

        Parameters
        __________
        ax : None or matplotlib AxesSubplot instance

        Returns
        _______
        matpliblib Figure instance, matplotlib AxesSubplot instance

        """
        if ax is None:
            fig, ax = plt.subplots()
            return fig, ax
        elif type(ax) is matplotlib.axes._subplots.AxesSubplot:
            return plt.gcf(), ax
        else:
            raise TypeError('ax agument must be None or an axis instance.')

    def get_grid(self, NFP, **kwargs):
        """Get grid for plotting.

        Parameters
        __________
        NFP : int
            number of (?)
        kwargs
            any arguments taken by LinearGrid (Default L=100, M=1, N=1)

        Returns
        _______
        LinearGrid

        """
        grid_args = {'rho':1.0, 'L':100, 'theta':0.0, 'M':1, 'zeta':0.0, 'N':1,
            'endpoint':False, 'NFP':NFP}
        for key in kwargs.keys():
            if key in grid_args.keys():
                grid_args[key] = kwargs[key]
        plot_axes = [0,1,2]
        grid_args['rho'] = self.__format_rtz__(grid_args['rho'])
        if grid_args['L'] == 1:
            plot_axes.remove(0)
        grid_args['theta'] = self.__format_rtz__(grid_args['theta'])
        if grid_args['M'] == 1:
            plot_axes.remove(1)
        grid_args['zeta'] = self.__format_rtz__(grid_args['zeta'])
        if grid_args['N'] == 1:
            plot_axes.remove(2)
        return LinearGrid(**grid_args), tuple(plot_axes)

    def plot_1d(self, eq, name, grid=None, ax=None, **kwargs):
        """Plot 1D slice from Equilibrium or Configuration.

        Parameters
        __________
        eq : Equilibrium or Configuration
            object from which to plot
        name : str
            name of variable to plot
        grid : Grid (optional)
            grid object defining coordinates to plot on
        ax : matplotlib AxesSubplot (optional)
            axis to plot on
        kwargs
            any arguments taken by LinearGrid (Default L=100, M=1, N=1)

        Returns
        _______
        axis

        """
        if grid is None:
            grid, plot_axis= self.get_grid(eq.NFP, **kwargs)
        if len(plot_axis) != 1:
            return ValueError('Grid must be 1D.')
        plot_axis=plot_axis[0]
        #dim = self.find_plot_ax_1d(grid)
        #theslice = self.grid_slice_1d(grid, dim)
        name_dict = self.format_name(name)
        ary = self.compute(eq, name_dict, grid)
        ax = self.format_ax(ax)
        ax.plot(grid.nodes[:,plot_axis], ary)
        ax.set_xlabel(self.axis_labels[plot_axis])
        ax.set_ylabel(self.name_label(name_dict))
        return ax

    def plot_2d(self, eq, name, grid=None, ax=None, **kwargs):
        """Plot 2D slice from Equilibrium or Configuration.

        Parameters
        __________
        eq : Equilibrium or Configuration
            object from which to plot
        name : str
            name of variable to plot
        grid : Grid (optional)
            grid object defining coordinates to plot on
        ax : matplotlib AxesSubplot (optional)
            axis to plot on
        kwargs
            any arguments taken by LinearGrid (Default L=100, M=100, N=1)

        Returns
        _______
        axis

        """
        if grid is None:
            if kwargs == {}:
                kwargs.update({'M':100})
            grid, plot_axes = self.get_grid(eq.NFP, **kwargs)
        if len(plot_axes) != 2:
            return ValueError('Grid must be 2D.')
        #dim = self.find_plot_ax_2d(grid)
        #theslice = self.grid_slice_2d(grid, dim)
        name_dict = self.format_name(name)
        ary = self.compute(eq, name_dict, grid)
        fig, ax = self.format_ax(ax)
        divider = make_axes_locatable(ax)
        #unroll array to be 2D
        if 0 in plot_axes:
            if 1 in plot_axes:
                sqary = np.zeros((grid.L, grid.M))
                for i in range(grid.M):
                    sqary[i,:] = ary[i*grid.L:(i+1)*grid.L]
            elif 2 in plot_axes:
                sqary = np.zeros((grid.L, grid.N))
                for i in range(grid.N):
                    sqary[i,:] = ary[i*grid.L:(i+1)*grid.L]
            else:
                raise ValueError('Grid must be 2D')
        elif 1 in plot_axes:
            sqary = np.zeros((grid.M, grid.N))
            for i in range(grid.M):
                sqary[i,:] = ary[i*grid.M:(i+1)*grid.N]
        else:
            raise ValueError('Grid must be 2D.')
        imshow_kwargs = {'origin'       : 'lower',
                        'interpolation' : 'bilinear',
                        'aspect'        : 'auto'}
        imshow_kwargs['extent'] = [grid.nodes[0,plot_axes[0]],
                grid.nodes[-1,plot_axes[0]], grid.nodes[0,plot_axes[1]],
                grid.nodes[-1,plot_axes[1]]]
        im = ax.imshow(sqary.T, **imshow_kwargs)
        cax_kwargs = {'size': '5%',
                    'pad'   : 0.05}
        cax = divider.append_axes('right', **cax_kwargs)
        cbar = fig.colorbar(im, cax=cax)
        cbar.formatter.set_powerlimits((0,0))
        cbar.update_ticks()
        ax.set_xlabel(self.axis_labels[plot_axes[0]])
        ax.set_ylabel(self.axis_labels[plot_axes[1]])
        ax.set_title(self.name_label(name_dict))
        return ax

    def plot_3dsurf(self):
        pass

    def compute(self, eq, name, grid):
        """Compute value specified by name on grid for equilibrium eq.

        Parameters
        __________
        eq : Configuration or Equilibrium
            Configuration or Equilibrium instance
        name : str or dict
            formatted string or parsed dictionary from format_name method
        grid : Grid
            grid on which to compute value specified by name

        Returns
        _______
        array of values

        """
        if type(name) is not dict:
            name_dict = self.format_name(name)
        else:
            name_dict = name
        # compute primitives from equilibtrium methods
        if name_dict['base'] == 'B':
            out = eq.compute_magnetic_field(grid)[self.__name_key__(name_dict)]
        elif name_dict['base'] == 'J':
            out = eq.compute_plasma_current(grid)[self.__name_key__(name_dict)]
        elif name_dict['base'] == '|B|':
            out = eq.compute_magnetic_field_magnitude(grid)[self.__name_key__(name_dict)]
        elif name_dict['base'] == '|F|':
            out = eq.compute_force_magnitude(grid)[self.__name_key__(name_dict)]
        else:
            raise NotImplementedError("No output for base named '{}'.".format(name_dict['base']))

        #secondary calculations
        power = name_dict['power']
        if power != '':
            try:
                power = float(power)
            except ValueError:
                #handle fractional exponents
                if '/' in power:
                    frac = power.split('/')
                    power = frac[0] / frac[1]
                else:
                    raise ValueError("Could not convert string to float: '{}'".format(power))
            out = out**power
        return out

    def format_name(self, name):
        """Parse name string into dictionary.

        Parameters
        __________
        name : str

        Returns
        _______
        parsed name : dict

        """
        name_dict = {'base':'', 'sups':'', 'subs':'', 'power':'', 'd':''}
        if '**' in name:
            parsename, power = name.split('**')
            if '_' in power or '^' in power:
                raise SyntaxError('Power operands must come after components and derivatives.')
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
        __________
        name_dict : dict
            name dictionary created by format_name method

        Returns
        _______
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
            #label = r'$' + name_dict['base'] + '^{' + esc + name_dict['sups'] +\
            #        ' ' + power + '}_{' + esc + name_dict['subs'] + '}$'

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
        #else:
        #    if name_dict['power'] == '':
        #        label = r'$d' + name_dict['base'] + '^{' + esc +\
        #            name_dict['sups'] + '}_{' + esc + name_dict['subs'] + '}/d'
        #            + name_dict['d'] + '$'
        #    else:
        #        label = r'$(d' + name_dict['base'] + '^{' + esc +\
        #            name_dict['sups'] + '}_{' + esc + name_dict['subs'] +\
        #            '})^{' + name_dict['power'] + '}$'
        label = r'$' + dstr0 + base + supstr + substr + dstr1 + '$'
        return label

    def __name_key__(self, name_dict):
        """Reconstruct name for dictionary key used in Configuration compute methods.

        Parameters
        __________
        name_dict : dict
            name dictionary created by format_name method

        Returns
        _______
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


def print_coeffs(cR, cZ, cL, zern_idx, lambda_idx):
    """prints coeffs to the terminal

    Parameters
    ----------
    cR,cZ,cL :
        spectral coefficients for R, Z, and lambda
    zern_idx, lambda_idx :
        mode numbers for zernike and fourier spectral basis.

    Returns
    -------

    """

    print('Fourier-Zernike coefficients:')
    for k, lmn in enumerate(zern_idx):
        print('l: {:3d}, m: {:3d}, n: {:3d}, cR: {:16.8e}, cZ: {:16.8e}'.format(
            lmn[0], lmn[1], lmn[2], cR[k], cZ[k]))

    print('Lambda coefficients')
    for k, mn in enumerate(lambda_idx):
        print('m: {:3d}, n: {:3d}, cL: {:16.8e}'.format(mn[0], mn[1], cL[k]))


def plot_coeffs(cR, cZ, cL, zern_idx, lambda_idx, cR_init=None, cZ_init=None, cL_init=None, **kwargs):
    """Scatter plots of zernike and lambda coefficients, before and after solving

    Parameters
    ----------
    cR : ndarray
        spectral coefficients of R
    cZ : ndarray
        spectral coefficients of Z
    cL : ndarray
        spectral coefficients of lambda
    zern_idx : ndarray
        array of (l,m,n) indices for each spectral R,Z coeff
    lambda_idx : ndarray
        indices for lambda spectral basis, ie an array of [m,n] for each spectral coefficient
    cR_init : ndarray
        initial spectral coefficients of R (Default value = None)
    cZ_init : ndarray
        initial spectral coefficients of Z (Default value = None)
    cL_init : ndarray
        initial spectral coefficients of lambda (Default value = None)
    **kwargs :
        additional plot formatting parameters

    Returns
    -------
    fig : matplotlib.figure
        handle to the figure
    ax : ndarray of matplotlib.axes
        handle to axes

    """
    nRZ = len(cR)
    nL = len(cL)
    fig, ax = plt.subplots(1, 3, figsize=(cR.size//5, 6))
    ax = ax.flatten()

    ax[0].scatter(cR, np.arange(nRZ), s=2, label='Final')
    if cR_init is not None:
        ax[0].scatter(cR_init, np.arange(nRZ), s=2, label='Init')
    ax[0].set_yticks(np.arange(nRZ))
    ax[0].set_yticklabels([str(i) for i in zern_idx])
    ax[0].set_xlabel('$R$')
    ax[0].set_ylabel('[l,m,n]')
    ax[0].axvline(0, c='k', lw=.25)
    ax[0].legend(loc='upper right')

    ax[1].scatter(cZ, np.arange(nRZ), s=2, label='Final')
    if cZ_init is not None:
        ax[1].scatter(cZ_init, np.arange(nRZ), s=2, label='Init')
    ax[1].set_yticks(np.arange(nRZ))
    ax[1].set_yticklabels([str(i) for i in zern_idx])
    ax[1].set_xlabel('$Z$')
    ax[1].set_ylabel('[l,m,n]')
    ax[1].axvline(0, c='k', lw=.25)
    ax[1].legend()

    ax[2].scatter(cL, np.arange(nL), s=2, label='Final')
    if cL_init is not None:
        ax[2].scatter(cL_init, np.arange(nL), s=2, label='Init')
    ax[2].set_yticks(np.arange(nL))
    ax[2].set_yticklabels([str(i) for i in lambda_idx])
    ax[2].set_xlabel('$\lambda$')
    ax[2].set_ylabel('[m,n]')
    ax[2].axvline(0, c='k', lw=.25)
    ax[2].legend()

    plt.subplots_adjust(wspace=.5)

    return fig, ax


def plot_coord_surfaces(cR, cZ, zern_idx, NFP, nr=10, nt=12, ax=None, bdryR=None, bdryZ=None, **kwargs):
    """Plots solutions (currently only zeta=0 plane)

    Parameters
    ----------
    cR : ndarray
        spectral coefficients of R
    cZ : ndarray
        spectral coefficients of Z
    zern_idx : ndarray
        indices for R,Z spectral basis, ie an array of [l,m,n] for each spectral coefficient
    NFP : int
        number of field periods
    nr : int
        number of flux surfaces to show (Default value = 10)
    nt : int
        number of theta lines to show (Default value = 12)
    ax : matplotlib.axes
        axes to plot on. If None, a new figure is created. (Default value = None)
    bdryR :
         R values of last closed flux surface (Default value = None)
    bdryZ :
         Z values of last closed flux surface (Default value = None)
    **kwargs :
        additional plot formatting parameters

    Returns
    -------
    ax : matplotlib.axes
        handle to axes used for the plot

    """

    Nr = 100
    Nt = 361
    rstep = Nr//nr
    tstep = Nt//nt
    zeta = kwargs.get('zeta', 0)
    r = np.linspace(0, 1, Nr)
    t = np.linspace(0, 2*np.pi, Nt)
    r, t = np.meshgrid(r, t, indexing='ij')
    r = r.flatten()
    t = t.flatten()
    z = zeta*np.ones_like(r)
    zernike_transform = ZernikeTransform([r, t, z], zern_idx, NFP)

    R = zernike_transform.transform(cR, 0, 0, 0).reshape((Nr, Nt))
    Z = zernike_transform.transform(cZ, 0, 0, 0).reshape((Nr, Nt))

    if ax is None:
        fig, ax = plt.subplots()
    # plot desired bdry
    if (bdryR is not None) and (bdryZ is not None):
        ax.plot(
            bdryR, bdryZ, color=colorblind_colors[1], lw=2, alpha=.5, dashes=(None, None))
    # plot r contours
    ax.plot(R.T[:, ::rstep], Z.T[:, ::rstep],
            color=colorblind_colors[0], lw=.5, dashes=(None, None))
    # plot actual bdry
    ax.plot(R.T[:, -1], Z.T[:, -1], color=colorblind_colors[0],
            lw=.5, dashes=(None, None))
    # plot theta contours
    ax.plot(R[:, ::tstep], Z[:, ::tstep],
            color=colorblind_colors[0], lw=.5, dashes=dashes[2])
    ax.axis('equal')
    ax.set_xlabel('$R$')
    ax.set_ylabel('$Z$')
    ax.set_title(kwargs.get('title'))
    return ax


def plot_IC(cR_init, cZ_init, zern_idx, NFP, nodes, cP, cI, **kwargs):
    """Plot initial conditions, such as the initial guess for flux surfaces,
    node locations, and profiles.

    Parameters
    ----------
    cR_init : ndarray
        spectral coefficients of R
    cZ_init : ndarray
        spectral coefficients of Z
    zern_idx : ndarray
        array of (l,m,n) indices for each spectral R,Z coeff
    NFP : int
        number of field periods
    nodes : ndarray
        locations of nodes in SFL coordinates
    cI : array-like
        paramters to pass to rotational transform function
    cP : array-like
        parameters to pass to pressure function
    **kwargs :
        additional plot formatting parameters


    Returns
    -------
    fig : matplotlib.figure
        handle to figure used for plotting
    ax : ndarray of matplotlib.axes
        handles to axes used for plotting

    """

    fig = plt.figure(figsize=kwargs.get('figsize', (9, 3)))
    gs = matplotlib.gridspec.GridSpec(2, 3)
    ax0 = plt.subplot(gs[:, 0])
    ax1 = plt.subplot(gs[:, 1], projection='polar')
    ax2 = plt.subplot(gs[0, 2])
    ax3 = plt.subplot(gs[1, 2])

    # coordinate surfaces
    plot_coord_surfaces(cR_init, cZ_init, zern_idx, NFP, nr=10, nt=12, ax=ax0)
    ax0.axis('equal')
    ax0.set_title(r'Initial guess, $\zeta=0$ plane')

    # node locations
    ax1.scatter(nodes[1], nodes[0], s=1)
    ax1.set_ylim(0, 1)
    ax1.set_xticks([0, np.pi/4, np.pi/2, 3/4*np.pi,
                    np.pi, 5/4*np.pi, 3/2*np.pi, 7/4*np.pi])
    ax1.set_xticklabels(['$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$',
                         r'$\pi$', r'$\frac{4\pi}{4}$', r'$\frac{3\pi}{2}$', r'$2\pi$'])
    ax1.set_yticklabels([])
    ax1.set_title(r'Node locations, $\zeta=0$ plane', pad=20)

    # profiles
    xx = np.linspace(0, 1, 100)
    ax2.plot(xx, presfun(xx, 0, cP), lw=1)
    ax2.set_ylabel(r'$p$')
    ax2.set_xticklabels([])
    ax2.set_title('Profiles')
    ax3.plot(xx, iotafun(xx, 0, cI), lw=1)
    ax3.set_ylabel(r'$\iota$')
    ax3.set_xlabel(r'$\rho$')
    plt.subplots_adjust(wspace=0.5, hspace=0.3)
    ax = np.array([ax0, ax1, ax2, ax3])

    return fig, ax


def plot_fb_err(equil, domain='real', normalize='local', log=True, cmap='plasma', **kwargs):
    """Plots force balance error

    Parameters
    ----------
    equil : dict
        dictionary of equilibrium solution quantities
    domain : str
        one of 'real', 'sfl'. What basis to use for plotting,
        real (R,Z) coordinates or straight field line (rho,vartheta) (Default value = 'real')
    normalize : str
        Whether and how to normalize values
        None, False - no normalization, values plotted are force error in Newtons/m^3
        'local' - normalize by local pressure gradient
        'global' - normalize by pressure gradient at rho=0.5
        True - same as 'global' (Default value = 'local')
    log : bool
        plot logarithm of error or absolute value (Default value = True)
    cmap : str
        colormap to use (Default value = 'plasma')
    **kwargs :
        additional plot formatting parameters

    Returns
    -------

    """

    cR = equil['cR']
    cZ = equil['cZ']
    cP = equil['cP']
    cI = equil['cI']
    Psi_lcfs = equil['Psi_lcfs']
    NFP = equil['NFP']
    zern_idx = equil['zern_idx']

    if np.max(zern_idx[:, 2]) == 0:
        Nz = 1
        rows = 1
    else:
        Nz = 6
        rows = 2

    Nr = kwargs.get('Nr', 51)
    Nv = kwargs.get('Nv', 90)
    Nlevels = kwargs.get('Nlevels', 100)

    nodes, vols = get_nodes_grid(NFP, nr=Nr, nt=Nv, nz=Nz)
    derivatives = get_needed_derivatives('all')
    zernike_transform = ZernikeTransform(nodes, zern_idx, NFP, derivatives)

    # compute fields components
    coord_der = compute_coordinate_derivatives(cR, cZ, zernike_transform)
    cov_basis = compute_covariant_basis(coord_der, zernike_transform)
    jacobian = compute_jacobian(coord_der, cov_basis, zernike_transform)
    con_basis = compute_contravariant_basis(
        coord_der, cov_basis, jacobian, zernike_transform)
    magnetic_field = compute_magnetic_field(cov_basis, jacobian, cI,
                                            Psi_lcfs, zernike_transform)
    plasma_current = compute_plasma_current(coord_der, cov_basis,
                                            jacobian, magnetic_field, cI, Psi_lcfs, zernike_transform)
    force_magnitude, p_mag = compute_force_magnitude(
        coord_der, cov_basis, con_basis, jacobian, magnetic_field, plasma_current, cP, cI, Psi_lcfs, zernike_transform)

    if domain == 'real':
        xlabel = r'R'
        ylabel = r'Z'
        R = zernike_transform.transform(cR, 0, 0, 0).reshape((Nr, Nv, Nz))
        Z = zernike_transform.transform(cZ, 0, 0, 0).reshape((Nr, Nv, Nz))
    elif domain == 'sfl':
        xlabel = r'$\vartheta$'
        ylabel = r'$\rho$'
        R = nodes[1].reshape((Nr, Nv, Nz))
        Z = nodes[0].reshape((Nr, Nv, Nz))
    else:
        raise ValueError(
            TextColors.FAIL + "domain must be either 'real' or 'sfl'" + TextColors.ENDC)

    if normalize == 'local':
        label = r'||F||/$\nabla$p'
        norm_errF = force_magnitude / p_mag
    elif normalize == 'global':
        label = r'||F||/$\nabla$p($\rho$=0.5)'
        halfn = np.where(nodes[0] == 0.5)[0][0]
        norm_errF = force_magnitude / p_mag[halfn]
    else:
        label = r'||F||'
        norm_errF = force_magnitude

    if log:
        label = r'$\mathregular{log}_{10}$('+label+')'
        norm_errF = np.log10(norm_errF)

    norm_errF = norm_errF.reshape((Nr, Nv, Nz))

    plt.figure()
    for k in range(Nz):
        ax = plt.subplot(rows, int(Nz/rows), k+1)
        cf = ax.contourf(R[:, :, k], Z[:, :, k], norm_errF[:, :, k],
                         cmap=cmap, extend='both', levels=Nlevels)
        if domain == 'real':
            ax.axis('equal')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        cbar = plt.colorbar(cf)
        if k == 0:
            cbar.ax.set_ylabel(label)
    plt.show()


def plot_comparison(equil0, equil1, label0='x0', label1='x1', **kwargs):
    """Plots force balance error

    Parameters
    ----------
    equil0, equil1 : dict
        dictionary of two equilibrium solution quantities
    label0, label1 : str
        labels for each equilibria
    **kwargs :
        additional plot formatting parameters

    Returns
    -------

    """

    cR0 = equil0.cR
    cZ0 = equil0.cZ
    NFP0 = equil0.NFP
    R_basis0 = equil0.R_basis
    Z_basis0 = equil0.Z_basis

    cR1 = equil1.cR
    cZ1 = equil1.cZ
    NFP1 = equil1.NFP
    R_basis1 = equil1.R_basis
    Z_basis1 = equil1.Z_basis

    if NFP0 == NFP1:
        NFP = NFP0
    else:
        raise ValueError(
            TextColors.FAIL + "NFP must be the same for both solutions" + TextColors.ENDC)

    if max(np.max(R_basis0.modes[:, 2]), np.max(R_basis1.modes[:, 2])) == 0:
        Nz = 1
        rows = 1
    else:
        Nz = 6
        rows = 2

    Nr = kwargs.get('Nr', 8)
    Nt = kwargs.get('Nt', 13)

    NNr = 100
    NNt = 360

    # constant rho surfaces
    grid_r = LinearGrid(L=Nr, M=NNt, N=Nz, NFP=NFP, endpoint=True)
    R_transf_0r = Transform(grid_r, R_basis0)
    Z_transf_0r = Transform(grid_r, Z_basis0)
    R_transf_1r = Transform(grid_r, R_basis1)
    Z_transf_1r = Transform(grid_r, Z_basis1)

    # constant theta surfaces
    grid_t = LinearGrid(L=NNr, M=Nt, N=Nz, NFP=NFP, endpoint=True)
    R_transf_0t = Transform(grid_t, R_basis0)
    Z_transf_0t = Transform(grid_t, Z_basis0)
    R_transf_1t = Transform(grid_t, R_basis1)
    Z_transf_1t = Transform(grid_t, Z_basis1)

    R0r = R_transf_0r.transform(cR0).reshape((Nr, NNt, Nz), order='F')
    Z0r = Z_transf_0r.transform(cZ0).reshape((Nr, NNt, Nz), order='F')
    R1r = R_transf_1r.transform(cR1).reshape((Nr, NNt, Nz), order='F')
    Z1r = Z_transf_1r.transform(cZ1).reshape((Nr, NNt, Nz), order='F')

    R0v = R_transf_0t.transform(cR0).reshape((NNr, Nt, Nz), order='F')
    Z0v = Z_transf_0t.transform(cZ0).reshape((NNr, Nt, Nz), order='F')
    R1v = R_transf_1t.transform(cR1).reshape((NNr, Nt, Nz), order='F')
    Z1v = Z_transf_1t.transform(cZ1).reshape((NNr, Nt, Nz), order='F')

    plt.figure()
    for k in range(Nz):
        ax = plt.subplot(rows, int(Nz/rows), k+1)

        ax.plot(R0r[0, 0, k], Z0r[0, 0, k], 'bo')
        s0 = ax.plot(R0r[:, :, k].T, Z0r[:, :, k].T, 'b-')
        ax.plot(R0v[:, :, k], Z0v[:, :, k], 'b:')

        ax.plot(R1r[0, 0, k], Z1r[0, 0, k], 'ro')
        s1 = ax.plot(R1r[:, :, k].T, Z1r[:, :, k].T, 'r-')
        ax.plot(R1v[:, :, k], Z1v[:, :, k], 'r:')

        ax.axis('equal')
        ax.set_xlabel('R')
        ax.set_ylabel('Z')
        if k == 0:
            s0[0].set_label(label0)
            s1[0].set_label(label1)
            ax.legend(fontsize='xx-small')
    plt.show()


def plot_vmec_comparison(vmec_data, equil):
    """Plots comparison of VMEC and DESC solutions

    Parameters
    ----------
    vmec_data : dict
        dictionary of VMEC solution quantities.
    equil : dict
        dictionary of DESC equilibrium solution quantities.

    Returns
    -------

    """

    cR = equil.cR
    cZ = equil.cZ
    NFP = equil.NFP
    R_basis = equil.R_basis
    Z_basis = equil.Z_basis

    Nr = 8
    Nt = 360
    if np.max(R_basis.modes[:, 2]) == 0:
        Nz = 1
        rows = 1
    else:
        Nz = 6
        rows = 2

    Nr_vmec = vmec_data['rmnc'].shape[0]-1
    s_idx = Nr_vmec % np.floor(Nr_vmec/(Nr-1))
    idxes = np.linspace(s_idx, Nr_vmec, Nr).astype(int)
    if s_idx != 0:
        idxes = np.pad(idxes, (1, 0), mode='constant')
        Nr += 1
    rho = np.sqrt(idxes/Nr_vmec)
    grid = LinearGrid(L=Nr, M=Nt, N=Nz, NFP=NFP, rho=rho, endpoint=True)
    R_transf = Transform(grid, R_basis)
    Z_transf = Transform(grid, Z_basis)

    R_desc = R_transf.transform(cR).reshape((Nr, Nt, Nz), order='F')
    Z_desc = Z_transf.transform(cZ).reshape((Nr, Nt, Nz), order='F')

    R_vmec, Z_vmec = vmec_interpolate(
        vmec_data['rmnc'][idxes], vmec_data['zmns'][idxes], vmec_data['xm'], vmec_data['xn'],
        np.unique(grid.nodes[:, 1]), np.unique(grid.nodes[:, 2]))

    plt.figure()
    for k in range(Nz):
        ax = plt.subplot(rows, int(Nz/rows), k+1)
        ax.plot(R_vmec[0, 0, k], Z_vmec[0, 0, k], 'bo')
        s_vmec = ax.plot(R_vmec[:, :, k].T, Z_vmec[:, :, k].T, 'b-')
        ax.plot(R_desc[0, 0, k], Z_desc[0, 0, k], 'ro')
        s_desc = ax.plot(R_desc[:, :, k].T, Z_desc[:, :, k].T, 'r--')
        ax.axis('equal')
        ax.set_xlabel('R')
        ax.set_ylabel('Z')
        if k == 0:
            s_vmec[0].set_label('VMEC')
            s_desc[0].set_label('DESC')
            ax.legend(fontsize='xx-small')
    plt.show()


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
