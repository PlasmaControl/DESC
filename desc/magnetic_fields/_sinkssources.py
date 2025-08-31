"""Classes for magnetic fields."""

import os
import numpy as np

from desc.basis import DoubleFourierSeries
from desc.compute import compute as compute_fun
from desc.compute.utils import get_params, get_transforms
from desc.grid import LinearGrid, Grid, _Grid
from desc.io import IOAble
from desc.optimizable import Optimizable, optimizable_parameter
from desc.utils import copy_coeffs, setdefault
#from desc.utils import cross, dot


class SinksSources(Optimizable, IOAble):
    """A magnetic field with perfect omnigenity (but is not necessarily analytic).

    Uses parameterization from Dudt et. al. [1]_

    Parameters
    ----------
    p_M : int
        Resolution of the Fourier series in eta for the omnigenity parameters.
    p_N : int
        Resolution of the Fourier series in alpha for the omnigenity parameters.
    NFP : int
        Number of field periods.
    x_mn : ndarray, optional
        Omnigenity parameters describing h(ρ,η,α). The coefficients correspond to the
        modes in `x_basis`. If not supplied, `x_mn` defaults to zero for all modes.

    Notes
    -----
    Doesn't conform to MagneticField API, as it only knows about :math:`|B|` in
    computational coordinates, not vector B in lab coordinates.

    References
    ----------
    .. [1] Dudt, Daniel W., et al. "Magnetic fields with general omnigenity."
       Journal of Plasma Physics (2024) doi:10.1017/S0022377824000151
    """

    _io_attrs_ = [
        "_p_M",
        "_p_N",
        "_NFP",
        "_x_basis",
        "_x_mn",
    ]

    _static_attrs = Optimizable._static_attrs + [
        "_p_M",
        "_p_N",
    ]

    def __init__(
        self,
        p_M=0,
        p_N=0,
        NFP=1,
        x_mn=None,
    ):
        self._p_M = int(p_M)
        self._p_N = int(p_N)
        self._NFP = int(NFP)
        self._x_basis = DoubleFourierSeries(
            M=self.p_M,
            N=self.p_N,
            NFP=self.NFP,
        )
        
        #self._x_basis = DoubleFourierSeries(
        #    M=self.p_M,
        #    N=self.p_N,
        #    NFP=self.NFP,
            # sym="cos(t)",
        #)

        if x_mn is None:
            self._x_mn = np.zeros(self.x_basis.num_modes)
        else:
            assert len(x_mn) == self.x_basis.num_modes
            self._x_mn = x_mn
            
        #if x_mn is None:
        #    self._x_mn = (self._p_M * 2 + 1) * (self._p_N * 2 + 1)#np.zeros(self.x_basis.num_modes)
        #else:
        #    assert len(x_mn) == (self._p_M * 2 + 1) * (self._p_N * 2 + 1) #self.x_basis.num_modes
        #    self._x_mn = x_mn

    def change_resolution(
        self,
        p_M=None,
        p_N=None,
        NFP=None,
    ):
        """Set the spectral resolution of field parameters.

        Parameters
        ----------
        p_M : int
            Resolution of the Fourier series in eta for the omnigenity params.
        p_N : int
            Resolution of the Fourier series in alpha for the omnigenity params.
        NFP : int
            Number of field periods.

        """
        self._NFP = setdefault(NFP, self.NFP)
        self._p_M = setdefault(p_M, self.p_M)
        self._p_N = setdefault(p_N, self.p_N)

        # change mapping parameters and basis
        #old_modes_map = #self.x_basis.modes
        #self.x_basis.change_resolution(
        #    self.p_M,
        #    self.p_N,
        #    NFP=self.NFP,  # sym="cos(t)"
        #)
        #self._x_mn = copy_coeffs(self.x_mn, old_modes_map, self.x_basis.modes)

    def compute(
        self,
        names,
        grid=None,
        params=None,
        transforms=None,
        profiles=None,
        data=None,
        **kwargs,
    ):
        """Compute the quantity given by name on grid.

        Parameters
        ----------
        names : str or array-like of str
            Name(s) of the quantity(s) to compute.
        grid : Grid, optional
            Grid of coordinates to evaluate at. The grid nodes are given in the usual
            (ρ,θ,ζ) coordinates, but θ is mapped to η and ζ is mapped to α.
            Defaults to a linearly space grid on the rho=1 surface.
        params : dict of ndarray
            Parameters from the equilibrium, such as R_lmn, Z_lmn, i_l, p_l, etc
            Defaults to attributes of self.
        transforms : dict of Transform
            Transforms for R, Z, lambda, etc. Default is to build from grid
        profiles : dict of Profile
            Profile objects for pressure, iota, current, etc. Defaults to attributes
            of self
        data : dict of ndarray
            Data computed so far, generally output from other compute functions
        **kwargs : dict, optional
            Valid keyword arguments are:

            * ``iota``: rotational transform
            * ``helicity``: helicity (defaults to self.helicity)

        Returns
        -------
        data : dict of ndarray
            Computed quantity and intermediate variables.

        """
        if isinstance(names, str):
            names = [names]
        if grid is None:
            grid = LinearGrid(
                theta=2 * self.M_B, N=2 * self.N_x, NFP=self.NFP, sym=False
            )
        elif not isinstance(grid, _Grid):
            raise TypeError(
                "must pass in a Grid object for argument grid!"
                f" instead got type {type(grid)}"
            )

        if params is None:
            params = get_params(names, obj=self, basis=kwargs.get("basis", "rpz"))
        if transforms is None:
            transforms = get_transforms(
                names,
                obj=self,
                grid=grid,
                jitable=kwargs.pop("jitable", False),
                **kwargs,
            )
        if data is None:
            data = {}
        profiles = {}

        data = compute_fun(
            self,
            names,
            params=params,
            transforms=transforms,
            profiles=profiles,
            data=data,
            #helicity=kwargs.pop("helicity", self.helicity),
            **kwargs,
        )
        return data

        # Assign the info of isothermaal coordinates

    #def save(self, file_name, file_format=None, file_mode="w"):
        #"""Save the object.
        #"""
        #file_name = os.path.expanduser(file_name)
        #raise OSError(
        #    "Saving CurrentPotentialField is not supported,"
        #    " as the potential function cannot be serialized."
        #)
        
    @property
    def p_M(self):
        """int: Number of (toroidal) sources."""
        return self._p_M

    @property
    def p_N(self):
        """int: Number of (poloidal) sources."""
        return self._p_N

    @property
    def NFP(self):
        """int: Number of (toroidal) field periods."""
        return self._NFP

    @property
    def x_basis(self):
        """ChebyshevDoubleFourierBasis: Spectral basis for x_mn."""
        return self._x_basis

    @optimizable_parameter
    @property
    def x_mn(self):
        """ndarray: Omnigenity coordinate mapping parameters."""
        return self._x_mn

    @x_mn.setter
    def x_mn(self, x_mn):
    #    assert len(x_mn) == len(self.x_mn)#.shape[0]
        self._x_mn = x_mn


#################################################################################
# Interpolate isothermal coordinates and interpolate on a different grid
def iso_coords_interp(tdata, _data, sgrid):

    # Temporary grid
    tgrid = LinearGrid(M=60, N=60)
    Phi = tdata["Phi_iso"]
    Psi = tdata["Psi_iso"]
    b0 = tdata["b_iso"]
    lamb_ratio = tdata["lambda_ratio"]

    _data["omega_1"] = tdata["omega_1"]
    _data["omega_2"] = tdata["omega_1"]

    _data["tau"] = tdata["tau"]
    _data["tau_1"] = tdata["tau_1"]
    _data["tau_2"] = tdata["tau_2"]

    lamb_u = tdata["lambda_iso_u"]
    lamb_v = tdata["lambda_iso_v"]

    # Data on plasma surface

    u_t = tdata["u_t"]
    u_z = tdata["u_z"]
    v_t = tdata["v_t"]
    v_z = tdata["v_z"]

    # Build new grids to allow interpolation between last grid points and theta = 2*pi or zeta = 2*pi
    m_size = tgrid.M * 2 + 1
    n_size = tgrid.N * 2 + 1

    # Rearrange variables
    # Add extra rows and columns to represent theta = 2pi or zeta = 2pi
    theta_mod = add_extra_coords(tdata["theta"], n_size, m_size, 0)
    zeta_mod = add_extra_coords(tdata["zeta"], n_size, m_size, 1)
    u_mod = zeta_mod - add_extra_periodic(Phi, n_size, m_size)
    v_mod = lamb_ratio * (
        theta_mod - add_extra_periodic(Psi, n_size, m_size) + b0 * u_mod
    )
    u_t_mod = add_extra_periodic(u_t, n_size, m_size)
    u_z_mod = add_extra_periodic(u_z, n_size, m_size)
    v_t_mod = add_extra_periodic(v_t, n_size, m_size)
    v_z_mod = add_extra_periodic(v_z, n_size, m_size)
    lamb_u_mod = add_extra_periodic(lamb_u, n_size, m_size)
    lamb_v_mod = add_extra_periodic(lamb_v, n_size, m_size)

    # Interpolate on theta_mod, zeta_mod
    points = jnp.array((zeta_mod.flatten(), theta_mod.flatten())).T

    # Interpolate isothermal coordinates
    _data["u_iso"] = interp2d(
        _data["theta"],
        _data["zeta"],
        theta_mod[:, 0],
        zeta_mod[0, :],
        u_mod,
        method="cubic",
    )
    _data["v_iso"] = interp2d(
        _data["theta"],
        _data["zeta"],
        theta_mod[:, 0],
        zeta_mod[0, :],
        v_mod,
        method="cubic",
    )

    _data["lambda_u"] = interp2d(
        _data["theta"],
        _data["zeta"],
        theta_mod[:, 0],
        zeta_mod[0, :],
        lamb_u_mod,
        method="cubic",
    )
    _data["lambda_v"] = interp2d(
        _data["theta"],
        _data["zeta"],
        theta_mod[:, 0],
        zeta_mod[0, :],
        lamb_v_mod,
        method="cubic",
    )

    # Interpolate derivatives of isothermal coordinates
    u0_t = interp2d(
        _data["theta"],
        _data["zeta"],
        theta_mod[:, 0],
        zeta_mod[0, :],
        u_t_mod,
        method="cubic",
    )
    u0_z = interp2d(
        _data["theta"],
        _data["zeta"],
        theta_mod[:, 0],
        zeta_mod[0, :],
        u_z_mod,
        method="cubic",
    )
    v0_t = interp2d(
        _data["theta"],
        _data["zeta"],
        theta_mod[:, 0],
        zeta_mod[0, :],
        v_t_mod,
        method="cubic",
    )
    v0_z = interp2d(
        _data["theta"],
        _data["zeta"],
        theta_mod[:, 0],
        zeta_mod[0, :],
        v_z_mod,
        method="cubic",
    )

    # Build harmonic vectors with interpolated data
    grad1 = (u0_t * _data["e^theta_s"].T + u0_z * _data["e^zeta_s"].T).T
    grad2 = (v0_t * _data["e^theta_s"].T + v0_z * _data["e^zeta_s"].T).T

    _data["e^u_s"] = grad1
    _data["e^v_s"] = grad2

    _data["e_u"] = (dot(grad1, grad1) ** (-1) * grad1.T).T
    _data["e_v"] = (dot(grad2, grad2) ** (-1) * grad2.T).T

    # Define the parameter "lambda" according to the paper
    _data["lambda_iso"] = dot(_data["e_u"], _data["e_u"]) ** (1 / 2)

    _data["w"] = comp_loc(_data["u_iso"], _data["v_iso"])

    return _data


# Load isothermal coordinates on the construction grid
def iso_coords_load(name, eq):

    # Temporary grid
    tgrid = LinearGrid(M=60, N=60)

    # Data on plasma surface
    tdata = eq.compute(["theta", "zeta"], grid=tgrid)

    tdata["u_iso"] = jnp.load(name + "u.npy")
    tdata["v_iso"] = jnp.load(name + "v.npy")
    tdata["Phi_iso"] = jnp.load(name + "Phi.npy")
    tdata["Psi_iso"] = jnp.load(name + "Psi.npy")
    tdata["b_iso"] = jnp.load(name + "b.npy")
    tdata["lambda_ratio"] = jnp.load(name + "ratio.npy")

    tdata["omega_1"] = jnp.load(name + "omega_1.npy")
    tdata["omega_2"] = jnp.load(name + "omega_2.npy")

    tdata["tau"] = jnp.load(name + "tau.npy")
    tdata["tau_1"] = jnp.load(name + "tau_1.npy")
    tdata["tau_2"] = jnp.load(name + "tau_2.npy")

    tdata["lambda_iso_u"] = jnp.load(name + "lambda_u.npy")
    tdata["lambda_iso_v"] = jnp.load(name + "lambda_v.npy")

    tdata["u_t"] = jnp.load(name + "u_t.npy")
    tdata["u_z"] = jnp.load(name + "u_z.npy")
    tdata["v_t"] = jnp.load(name + "v_t.npy")
    tdata["v_z"] = jnp.load(name + "v_z.npy")

    return tdata


def interp_grid(theta, zeta, w_surface, tdata):

    # Find grids for dipoles
    s_grid = alt_grid(theta, zeta)

    # Evaluate data on grids of dipoles
    s_data = w_surface.compute(
        [
            "theta",
            "zeta",
            "e^theta_s",
            "e^zeta_s",
            "x",
            "e_theta",  # extra vector needed for the poloidal wire contours
            '|e_theta x e_zeta|',
        ],
        grid=s_grid,
    )

    return iso_coords_interp(tdata, s_data, w_surface)


def add_extra(data_, n_size, m_size):

    _mod = data_.reshape((n_size, m_size)).T
    _mod = jnp.column_stack([_mod, _mod[0:m_size, 0]])
    _mod = jnp.vstack([_mod, 2 * jnp.pi * jnp.ones(_mod.shape[1])])

    return _mod


def add_extra_periodic(data_, n_size, m_size):

    _mod = data_.reshape((n_size, m_size)).T
    _mod = jnp.column_stack([_mod, _mod[:, 0]])
    _mod = jnp.vstack([_mod, _mod[0, :]])

    return _mod


def add_extra_coords(data_, n_size, m_size, ind):

    _mod = data_.reshape((n_size, m_size)).T
    # _mod = jnp.vstack( [ _mod, _mod[0:m_size,0] ] )

    if ind == 0:
        _mod = jnp.column_stack([_mod, _mod[:, 0]])
        _mod = jnp.vstack([_mod, 2 * jnp.pi * jnp.ones(_mod.shape[1])])

    if ind == 1:
        _mod = jnp.column_stack([_mod, 2 * jnp.pi * jnp.ones_like(_mod[:, 0])])
        _mod = jnp.vstack([_mod, _mod[0, :]])

    return _mod


def alt_grid(theta, zeta):

    theta_grid, zeta_grid = jnp.meshgrid(theta, zeta)
    theta_flat = theta_grid.flatten()
    zeta_flat = zeta_grid.flatten()

    return Grid(
        jnp.stack((jnp.ones_like(theta_flat), theta_flat, zeta_flat)).T, jitable=True
    )


def alt_grid_sticks(theta, zeta, sgrid):

    theta_grid, zeta_grid = jnp.meshgrid(theta, zeta)
    theta_flat = theta_grid.flatten()
    zeta_flat = zeta_grid.flatten()

    return Grid(
        jnp.stack((jnp.ones_like(theta_flat), theta_flat, zeta_flat)).T,
        weights=jnp.ones_like(theta_flat),
        NFP=sgrid.NFP,
        jitable=True,
    )


def densify_linspace(arr, points_per_interval=1):
    """
    Given a jnp.linspace array, return a new array with additional points
    between each pair of original points while keeping all original points.

    Args:
        arr (jnp.ndarray): Original 1D array (typically from jnp.linspace)
        points_per_interval (int): Number of points to insert between each pair

    Returns:
        jnp.ndarray: New array with original + additional interpolated points
    """
    if arr.ndim != 1:
        raise ValueError("Only 1D arrays supported")

    new_points = []
    for i in range(len(arr) - 1):
        start = arr[i]
        end = arr[i + 1]

        # Include original point
        new_points.append(start)

        # Generate internal points (excluding end to avoid duplication)
        if points_per_interval > 0:
            inter_points = jnp.linspace(start, end, points_per_interval + 2)[1:-1]
            new_points.append(inter_points)

    new_points.append(arr[-1])  # Don't forget the last point!

    return jnp.concatenate([jnp.atleast_1d(p) for p in new_points])