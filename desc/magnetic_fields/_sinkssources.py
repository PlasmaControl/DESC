"""Classes for magnetic fields."""

import os
import warnings
from abc import ABC, abstractmethod
from collections.abc import MutableSequence

import numpy as np
from diffrax import (
    DiscreteTerminatingEvent,
    ODETerm,
    PIDController,
    SaveAt,
    Tsit5,
    diffeqsolve,
)
from interpax import approx_df, interp1d, interp2d, interp3d
from netCDF4 import Dataset, chartostring, stringtochar
from scipy.constants import mu_0

from desc.backend import jit, jnp, sign
from desc.basis import (
    ChebyshevDoubleFourierBasis,
    ChebyshevPolynomial,
    DoubleFourierSeries,
)
from desc.batching import batch_map
from desc.compute import compute as compute_fun
from desc.compute.utils import get_params, get_transforms
from desc.derivatives import Derivative
from desc.equilibrium import EquilibriaFamily, Equilibrium
from desc.grid import LinearGrid, _Grid
from desc.integrals import compute_B_plasma
from desc.io import IOAble
from desc.optimizable import Optimizable, OptimizableCollection, optimizable_parameter
from desc.transform import Transform
from desc.utils import (
    copy_coeffs,
    dot,
    errorif,
    flatten_list,
    rpz2xyz,
    rpz2xyz_vec,
    safediv,
    setdefault,
    warnif,
    xyz2rpz,
    xyz2rpz_vec,
)
from desc.vmec_utils import ptolemy_identity_fwd, ptolemy_identity_rev

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
            #sym="cos(t)",
        )
        
        if x_mn is None:
            self._x_mn = np.zeros(self.x_basis.num_modes)
        else:
            assert len(x_mn) == self.x_basis.num_modes
            self._x_mn = x_mn

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
        old_L_B = self.L_B

        self._NFP = setdefault(NFP, self.NFP)
        self._p_M = setdefault(M_x, self.p_M)
        self._p_N = setdefault(N_x, self.p_N)

        # change mapping parameters and basis
        old_modes_map = self.x_basis.modes
        self.x_basis.change_resolution(
            self.p_M, self.p_N, NFP=self.NFP,# sym="cos(t)"
        )
        self._x_mn = copy_coeffs(self.x_mn, old_modes_map, self.x_basis.modes)

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
            helicity=kwargs.pop("helicity", self.helicity),
            **kwargs,
        )
        return data

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
        assert len(x_mn) == self.x_basis.num_modes
        self._x_mn = x_mn