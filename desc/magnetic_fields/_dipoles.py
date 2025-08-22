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


class Dipoles(Optimizable, IOAble):
    """A field generated with dipoles on the winding surface

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
        "_x_mn_poloidal",
        "_x_mn_toroidal",
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
        x_mn_poloidal = None,
        x_mn_toroidal = None,
    ):
        self._p_M = int(p_M)
        self._p_N = int(p_N)
        self._NFP = int(NFP)
        
        self._x_poloidal_basis = DoubleFourierSeries(
            M=self.p_M,
            N=self.p_N,
            NFP=self.NFP,
        )

        self._x_toroidal_basis = DoubleFourierSeries(
            M=self.p_M,
            N=self.p_N,
            NFP=self.NFP,
        )
        
        if x_mn_poloidal is None:
            self._x_mn_poloidal = np.zeros(self.x_poloidal_basis.num_modes)
        else:
            assert len(x_mn_poloidal) == self.x_poloidal_basis.num_modes
            self._x_mn_poloidal = x_mn_poloidal

        if x_mn_toroidal is None:
            self._x_mn_toroidal = np.zeros(self.x_toroidal_basis.num_modes)
        else:
            assert len(x_mn_toroidal) == self.x_toroidal_basis.num_modes
            self._x_mn_toroidal = x_mn_toroidal

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

    def save(self, file_name, file_format=None, file_mode="w"):
        """Save the object.

        **Not supported for this object!**

        Parameters
        ----------
        file_name : str file path OR file instance
            location to save object
        file_format : str (Default hdf5)
            format of save file. Only used if file_name is a file path
        file_mode : str (Default w - overwrite)
            mode for save file. Only used if file_name is a file path

        """
        file_name = os.path.expanduser(file_name)
        raise OSError(
            "Saving CurrentPotentialField is not supported,"
            " as the potential function cannot be serialized."
        )
        
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
    def x_poloidal_basis(self):
        """ChebyshevDoubleFourierBasis: Spectral basis for x_mn."""
        return self._x_poloidal_basis

    @property
    def x_toroidal_basis(self):
        """ChebyshevDoubleFourierBasis: Spectral basis for x_mn."""
        return self._x_poloidal_basis

    @optimizable_parameter
    @property
    def x_mn_poloidal(self):
        """ndarray: Omnigenity coordinate mapping parameters."""
        return self._x_mn_poloidal

    @x_mn_poloidal.setter
    def x_mn_poloidal(self, x_mn_poloidal):
    #    assert len(x_mn) == len(self.x_mn)#.shape[0]
        self._x_mn_poloidal = x_mn_poloidal

    @optimizable_parameter
    @property
    def x_mn_toroidal(self):
        """ndarray: Omnigenity coordinate mapping parameters."""
        return self._x_mn_toroidal

    @x_mn_toroidal.setter
    def x_mn_toroidal(self, x_mn_toroidal):
    #    assert len(x_mn) == len(self.x_mn)#.shape[0]
        self._x_mn_toroidal = x_mn_toroidal