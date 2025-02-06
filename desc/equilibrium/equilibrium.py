"""Core class representing MHD equilibrium."""

import copy
import numbers
import warnings
from collections.abc import MutableSequence

import numpy as np
from scipy import special
from scipy.constants import mu_0

from desc.backend import execute_on_cpu, jnp
from desc.basis import FourierZernikeBasis, fourier, zernike_radial
from desc.compat import ensure_positive_jacobian
from desc.compute import compute as compute_fun
from desc.compute import data_index
from desc.compute.utils import (
    _grow_seeds,
    get_data_deps,
    get_params,
    get_profiles,
    get_transforms,
)
from desc.geometry import (
    FourierRZCurve,
    FourierRZToroidalSurface,
    ZernikeRZToroidalSection,
)
from desc.grid import Grid, LinearGrid, QuadratureGrid, _Grid
from desc.input_reader import InputReader
from desc.io import IOAble
from desc.objectives import (
    ForceBalance,
    ObjectiveFunction,
    get_equilibrium_objective,
    get_fixed_axis_constraints,
    get_fixed_boundary_constraints,
)
from desc.optimizable import Optimizable, optimizable_parameter
from desc.optimize import Optimizer
from desc.perturbations import perturb
from desc.profiles import HermiteSplineProfile, PowerSeriesProfile, SplineProfile
from desc.transform import Transform
from desc.utils import (
    ResolutionWarning,
    check_nonnegint,
    check_posint,
    copy_coeffs,
    errorif,
    only1,
    setdefault,
    warnif,
)

from ..compute.data_index import is_0d_vol_grid, is_1dr_rad_grid, is_1dz_tor_grid
from .coords import get_rtz_grid, is_nested, map_coordinates, to_sfl
from .initial_guess import set_initial_guess
from .utils import parse_axis, parse_profile, parse_surface


class Equilibrium(IOAble, Optimizable):
    """Equilibrium is an object that represents a plasma equilibrium.

    It contains information about a plasma state, including the shapes of flux surfaces
    and profile inputs. It can compute additional information, such as the magnetic
    field and plasma currents, as well as "solving" itself by finding the equilibrium
    fields, and perturbing those fields to find nearby equilibria.

    Parameters
    ----------
    Psi : float (optional)
        total toroidal flux (in Webers) within LCFS. Default 1.0
    NFP : int (optional)
        number of field periods Default ``surface.NFP`` or 1
    L : int (optional)
        Radial resolution. Default 2*M for ``spectral_indexing=='fringe'``, else M
    M : int (optional)
        Poloidal resolution. Default surface.M or 1
    N : int (optional)
        Toroidal resolution. Default surface.N or 0
    L_grid : int (optional)
        resolution of real space nodes in radial direction
    M_grid : int (optional)
        resolution of real space nodes in poloidal direction
    N_grid : int (optional)
        resolution of real space nodes in toroidal direction
    pressure : Profile or ndarray shape(k,2) (optional)
        Pressure profile or array of mode numbers and spectral coefficients.
        Default is a PowerSeriesProfile with zero pressure
    iota : Profile or ndarray shape(k,2) (optional)
        Rotational transform profile or array of mode numbers and spectral coefficients
    current : Profile or ndarray shape(k,2) (optional)
        Toroidal current profile or array of mode numbers and spectral coefficients
        Default is a PowerSeriesProfile with zero toroidal current
    electron_temperature : Profile or ndarray shape(k,2) (optional)
        Electron temperature (eV) profile or array of mode numbers and spectral
        coefficients. Must be supplied with corresponding density.
        Cannot specify both kinetic profiles and pressure.
    electron_density : Profile or ndarray shape(k,2) (optional)
        Electron density (m^-3) profile or array of mode numbers and spectral
        coefficients. Must be supplied with corresponding temperature.
        Cannot specify both kinetic profiles and pressure.
    ion_temperature : Profile or ndarray shape(k,2) (optional)
        Ion temperature (eV) profile or array of mode numbers and spectral coefficients.
        Default is to assume electrons and ions have the same temperature.
    atomic_number : Profile or ndarray shape(k,2) (optional)
        Effective atomic number (Z_eff) profile or ndarray of mode numbers and spectral
        coefficients. Default is 1
    anisotropy : Profile or ndarray
        Anisotropic pressure profile or array of mode numbers and spectral coefficients.
        Default is a PowerSeriesProfile with zero anisotropic pressure.
    surface: Surface or ndarray shape(k,5) (optional)
        Fixed boundary surface shape, as a Surface object or array of
        spectral mode numbers and coefficients of the form [l, m, n, R, Z].
        Default is a FourierRZToroidalSurface with major radius 10 and minor radius 1
    axis : Curve or ndarray shape(k,3) (optional)
        Initial guess for the magnetic axis as a Curve object or ndarray
        of mode numbers and spectral coefficients of the form [n, R, Z].
        Default is the centroid of the surface.
    sym : bool (optional)
        Whether to enforce stellarator symmetry. Default surface.sym or False.
    spectral_indexing : str (optional)
        Type of Zernike indexing scheme to use. Default ``'ansi'``
    check_orientation : bool
        ensure that this equilibrium has a right handed orientation. Do not set to False
        unless you are sure the parameterization you have given is right handed
        (ie, e_theta x e_zeta points outward from the surface).
    ensure_nested : bool
        If True, and the default initial guess does not produce nested surfaces,
        run a small optimization problem to attempt to refine initial guess to improve
        coordinate mapping.

    """

    _io_attrs_ = [
        "_sym",
        "_R_sym",
        "_Z_sym",
        "_Psi",
        "_NFP",
        "_L",
        "_M",
        "_N",
        "_R_lmn",
        "_Z_lmn",
        "_L_lmn",
        "_R_basis",
        "_Z_basis",
        "_L_basis",
        "_surface",
        "_axis",
        "_pressure",
        "_iota",
        "_current",
        "_electron_temperature",
        "_electron_density",
        "_ion_temperature",
        "_atomic_number",
        "_anisotropy",
        "_spectral_indexing",
        "_bdry_mode",
        "_L_grid",
        "_M_grid",
        "_N_grid",
    ]

    @execute_on_cpu
    def __init__(
        self,
        Psi=1.0,
        NFP=None,
        L=None,
        M=None,
        N=None,
        L_grid=None,
        M_grid=None,
        N_grid=None,
        pressure=None,
        iota=None,
        current=None,
        electron_temperature=None,
        electron_density=None,
        ion_temperature=None,
        atomic_number=None,
        anisotropy=None,
        surface=None,
        axis=None,
        sym=None,
        spectral_indexing=None,
        check_orientation=True,
        ensure_nested=True,
        **kwargs,
    ):
        errorif(
            not isinstance(Psi, numbers.Real),
            ValueError,
            f"Psi should be a real integer or float, got {type(Psi)}",
        )
        self._Psi = float(Psi)

        errorif(
            spectral_indexing
            not in [
                None,
                "ansi",
                "fringe",
            ],
            ValueError,
            "spectral_indexing should be one of 'ansi', 'fringe', None, got "
            + f"{spectral_indexing}",
        )
        self._spectral_indexing = setdefault(
            spectral_indexing, getattr(surface, "spectral_indexing", "ansi")
        )

        NFP = check_posint(NFP, "NFP")
        self._NFP = int(
            setdefault(NFP, getattr(surface, "NFP", getattr(axis, "NFP", 1)))
        )

        # stellarator symmetry for bases
        errorif(
            sym
            not in [
                None,
                True,
                False,
            ],
            ValueError,
            f"sym should be one of True, False, None, got {sym}",
        )
        self._sym = bool(setdefault(sym, getattr(surface, "sym", False)))
        self._R_sym = "cos" if self.sym else False
        self._Z_sym = "sin" if self.sym else False

        # surface
        self._surface, self._bdry_mode = parse_surface(
            surface, self.NFP, self.sym, self.spectral_indexing
        )

        # magnetic axis
        self._axis = parse_axis(axis, self.NFP, self.sym, self.surface)

        # resolution
        L = check_nonnegint(L, "L")
        M = check_nonnegint(M, "M")
        N = check_nonnegint(N, "N")
        L_grid = check_nonnegint(L_grid, "L_grid")
        M_grid = check_nonnegint(M_grid, "M_grid")
        N_grid = check_nonnegint(N_grid, "N_grid")

        self._N = int(setdefault(N, self.surface.N))
        self._M = int(setdefault(M, self.surface.M))
        self._L = int(
            setdefault(
                L,
                max(
                    self.surface.L,
                    self.M if (self.spectral_indexing == "ansi") else 2 * self.M,
                ),
            )
        )
        self._L_grid = setdefault(L_grid, 2 * self.L)
        self._M_grid = setdefault(M_grid, 2 * self.M)
        self._N_grid = setdefault(N_grid, 2 * self.N)

        self._surface.change_resolution(self.L, self.M, self.N)
        self._axis.change_resolution(self.N)

        # bases
        self._R_basis = FourierZernikeBasis(
            L=self.L,
            M=self.M,
            N=self.N,
            NFP=self.NFP,
            sym=self._R_sym,
            spectral_indexing=self.spectral_indexing,
        )
        self._Z_basis = FourierZernikeBasis(
            L=self.L,
            M=self.M,
            N=self.N,
            NFP=self.NFP,
            sym=self._Z_sym,
            spectral_indexing=self.spectral_indexing,
        )
        self._L_basis = FourierZernikeBasis(
            L=self.L,
            M=self.M,
            N=self.N,
            NFP=self.NFP,
            sym=self._Z_sym,
            spectral_indexing=self.spectral_indexing,
        )

        # profiles
        self._pressure = None
        self._iota = None
        self._current = None
        self._electron_temperature = None
        self._electron_density = None
        self._ion_temperature = None
        self._atomic_number = None

        if current is None and iota is None:
            current = 0
        use_kinetic = any(
            [electron_temperature is not None, electron_density is not None]
        )
        errorif(
            current is not None and iota is not None,
            ValueError,
            "Cannot specify both iota and current profiles.",
        )
        errorif(
            ((pressure is not None) or (anisotropy is not None)) and use_kinetic,
            ValueError,
            "Cannot specify both pressure and kinetic profiles.",
        )
        errorif(
            use_kinetic and (electron_temperature is None or electron_density is None),
            ValueError,
            "Must give at least electron temperature and density to use "
            + "kinetic profiles.",
        )
        if use_kinetic and atomic_number is None:
            atomic_number = 1
        if use_kinetic and ion_temperature is None:
            ion_temperature = electron_temperature
        if not use_kinetic and pressure is None:
            pressure = 0

        self.electron_temperature = parse_profile(
            electron_temperature, "electron temperature"
        )
        self.electron_density = parse_profile(electron_density, "electron density")
        self.ion_temperature = parse_profile(ion_temperature, "ion temperature")
        self.atomic_number = parse_profile(atomic_number, "atomic number")
        self.pressure = parse_profile(pressure, "pressure")
        self.anisotropy = parse_profile(anisotropy, "anisotropy")
        self._iota = self._current = None
        self.iota = parse_profile(iota, "iota")
        self.current = parse_profile(current, "current")

        # ensure profiles have the right resolution
        for profile in [
            "pressure",
            "iota",
            "current",
            "electron_temperature",
            "electron_density",
            "ion_temperature",
            "atomic_number",
            "anisotropy",
        ]:
            p = getattr(self, profile)
            if hasattr(p, "change_resolution"):
                p.change_resolution(max(p.basis.L, self.L))
            warnif(
                isinstance(p, PowerSeriesProfile) and p.sym != "even",
                msg=f"{profile} profile is not an even power series.",
            )

        # ensure number of field periods agree before setting guesses
        eq_NFP = self.NFP
        surf_NFP = self.surface.NFP if hasattr(self.surface, "NFP") else self.NFP
        axis_NFP = self._axis.NFP
        errorif(
            not (eq_NFP == surf_NFP == axis_NFP),
            ValueError,
            "Unequal number of field periods for equilibrium "
            + f"{eq_NFP}, surface {surf_NFP}, and axis {axis_NFP}",
        )

        # make sure symmetry agrees
        errorif(
            self.sym != self.surface.sym,
            ValueError,
            "Surface and Equilibrium must have the same symmetry",
        )
        self._R_lmn = np.zeros(self.R_basis.num_modes)
        self._Z_lmn = np.zeros(self.Z_basis.num_modes)
        self._L_lmn = np.zeros(self.L_basis.num_modes)

        if ("R_lmn" in kwargs) or ("Z_lmn" in kwargs):
            assert ("R_lmn" in kwargs) and ("Z_lmn" in kwargs), "Must give both R and Z"
            self.R_lmn = kwargs.pop("R_lmn")
            self.Z_lmn = kwargs.pop("Z_lmn")
            self.L_lmn = kwargs.pop("L_lmn", jnp.zeros(self.L_basis.num_modes))
        else:
            self.set_initial_guess(ensure_nested=ensure_nested)
        if check_orientation:
            ensure_positive_jacobian(self)
        if kwargs.get("check_kwargs", True):
            errorif(
                len(kwargs),
                TypeError,
                f"Equilibrium got unexpected kwargs: {kwargs.keys()}",
            )

    def _set_up(self):
        """Set unset attributes after loading.

        To ensure object has all properties needed for current DESC version.
        Allows for backwards-compatibility with equilibria saved/ran with older
        DESC versions.
        """
        for attribute in self._io_attrs_:
            if not hasattr(self, attribute):
                setattr(self, attribute, None)

        if self.current is not None and hasattr(self.current, "_get_transform"):
            # Need to rebuild derivative matrices to get higher order derivatives
            # on equilibrium's saved before GitHub pull request #586.
            self.current._transform = self.current._get_transform(self.current.grid)

        # ensure things that should be ints are ints
        self._L = int(self._L)
        self._M = int(self._M)
        self._N = int(self._N)
        self._NFP = int(self._NFP)
        self._L_grid = int(self._L_grid)
        self._M_grid = int(self._M_grid)
        self._N_grid = int(self._N_grid)

    def _sort_args(self, args):
        """Put arguments in a canonical order. Returns unique sorted elements.

        For Equilibrium, alphabetical order seems to lead to some numerical instability
        so we enforce a particular order that has worked well.
        """
        arg_order = (
            "R_lmn",
            "Z_lmn",
            "L_lmn",
            "p_l",
            "i_l",
            "c_l",
            "Psi",
            "Te_l",
            "ne_l",
            "Ti_l",
            "Zeff_l",
            "a_lmn",
            "Ra_n",
            "Za_n",
            "Rb_lmn",
            "Zb_lmn",
            "I",
            "G",
            "Phi_mn",
        )
        assert sorted(args) == sorted(arg_order)
        return [arg for arg in arg_order if arg in args]

    def __repr__(self):
        """String form of the object."""
        return (
            type(self).__name__
            + " at "
            + str(hex(id(self)))
            + f" (L={self.L}, M={self.M}, N={self.N}, NFP={self.NFP},"
            + f" spectral_indexing={self.spectral_indexing})"
        )

    def set_initial_guess(self, *args, ensure_nested=True):
        """Set the initial guess for the flux surfaces, eg R_lmn, Z_lmn, L_lmn.

        Parameters
        ----------
        eq : Equilibrium
            Equilibrium to initialize
        args :
            either:
              - No arguments, in which case eq.surface will be scaled for the guess.
              - Another Surface object, which will be scaled to generate the guess.
                Optionally a Curve object may also be supplied for the magnetic axis.
              - Another Equilibrium, whose flux surfaces will be used.
              - File path to a VMEC or DESC equilibrium, which will be loaded and used.
              - Grid and 2-3 ndarrays, specifying the flux surface locations (R, Z, and
                optionally lambda) at fixed flux coordinates. All arrays should have the
                same length. Optionally, an ndarray of shape(k,3) may be passed instead
                of a grid.
        ensure_nested : bool
            If True, and the default initial guess does not produce nested surfaces,
            run a small optimization problem to attempt to refine initial guess to
            improve coordinate mapping.

        Examples
        --------
        Use existing equil.surface and scales down for guess:

        >>> equil.set_initial_guess()

        Use supplied Surface and scales down for guess. Assumes axis is centroid
        of user supplied surface:

        >>> equil.set_initial_guess(surface)

        Optionally, an interior surface may be scaled by giving the surface a
        flux label:

        >>> surf = FourierRZToroidalSurface(rho=0.7)
        >>> equil.set_initial_guess(surf)

        Use supplied Surface and a supplied Curve for axis and scales between
        them for guess:

        >>> equil.set_initial_guess(surface, curve)

        Use the flux surfaces from an existing Equilibrium:

        >>> equil.set_initial_guess(equil2)

        Use flux surfaces from existing Equilibrium or VMEC output stored on disk:

        >>> equil.set_initial_guess(path_to_saved_DESC_or_VMEC_output)

        Use flux surfaces specified by points:
        nodes should either be a Grid or an ndarray, shape(k,3) giving the locations
        in rho, theta, zeta coordinates. R, Z, and optionally lambda should be
        array-like, shape(k,) giving the corresponding real space coordinates

        >>> equil.set_initial_guess(nodes, R, Z, lambda)

        """
        set_initial_guess(self, *args, ensure_nested=ensure_nested)

    def copy(self, deepcopy=True):
        """Return a (deep)copy of this equilibrium."""
        if deepcopy:
            new = copy.deepcopy(self)
        else:
            new = copy.copy(self)
        return new

    @execute_on_cpu
    def change_resolution(
        self,
        L=None,
        M=None,
        N=None,
        L_grid=None,
        M_grid=None,
        N_grid=None,
        NFP=None,
        sym=None,
    ):
        """Set the spectral resolution and real space grid resolution.

        Parameters
        ----------
        L : int
            Maximum radial Zernike mode number.
        M : int
            Maximum poloidal Fourier mode number.
        N : int
            Maximum toroidal Fourier mode number.
        L_grid : int
            Radial real space grid resolution.
        M_grid : int
            Poloidal real space grid resolution.
        N_grid : int
            Toroidal real space grid resolution.
        NFP : int
            Number of field periods.
        sym : bool
            Whether to enforce stellarator symmetry.

        """
        warnif(
            L is not None and L < self.L,
            UserWarning,
            "Reducing radial (L) resolution can make plasma boundary inconsistent. "
            + "Recommend calling `eq.surface = eq.get_surface_at(rho=1.0)`",
        )
        self._L = int(setdefault(L, self.L))
        self._M = int(setdefault(M, self.M))
        self._N = int(setdefault(N, self.N))
        self._L_grid = int(setdefault(L_grid, self.L_grid))
        self._M_grid = int(setdefault(M_grid, self.M_grid))
        self._N_grid = int(setdefault(N_grid, self.N_grid))
        self._NFP = int(setdefault(NFP, self.NFP))
        self._sym = bool(setdefault(sym, self.sym))

        old_modes_R = self.R_basis.modes
        old_modes_Z = self.Z_basis.modes
        old_modes_L = self.L_basis.modes

        self.R_basis.change_resolution(
            self.L,
            self.M,
            self.N,
            NFP=self.NFP,
            sym="cos" if self.sym else self.sym,
        )
        self.Z_basis.change_resolution(
            self.L,
            self.M,
            self.N,
            NFP=self.NFP,
            sym="sin" if self.sym else self.sym,
        )
        self.L_basis.change_resolution(
            self.L,
            self.M,
            self.N,
            NFP=self.NFP,
            sym="sin" if self.sym else self.sym,
        )

        for profile in [
            "pressure",
            "iota",
            "current",
            "electron_temperature",
            "electron_density",
            "ion_temperature",
            "atomic_number",
            "anisotropy",
        ]:
            p = getattr(self, profile)
            if hasattr(p, "change_resolution"):
                p.change_resolution(max(p.basis.L, self.L))

        self.surface.change_resolution(
            self.L,
            self.M,
            self.N,
            NFP=self.NFP,
            sym=self.sym,
        )
        self.axis.change_resolution(
            self.N,
            NFP=self.NFP,
            sym=self.sym,
        )

        self._R_lmn = copy_coeffs(self.R_lmn, old_modes_R, self.R_basis.modes)
        self._Z_lmn = copy_coeffs(self.Z_lmn, old_modes_Z, self.Z_basis.modes)
        self._L_lmn = copy_coeffs(self.L_lmn, old_modes_L, self.L_basis.modes)

    @execute_on_cpu
    def get_surface_at(self, rho=None, theta=None, zeta=None):
        """Return a representation for a given coordinate surface.

        Parameters
        ----------
        rho, theta, zeta : float or None
            radial, poloidal, or toroidal coordinate for the surface. Only
            one may be specified.

        Returns
        -------
        surf : Surface
            object representing the given surface, either a FourierRZToroidalSurface
            for surfaces of constant rho, or a ZernikeRZToroidalSection for
            surfaces of constant zeta.

        """
        errorif(
            not only1(rho is not None, theta is not None, zeta is not None),
            ValueError,
            f"Only one coordinate can be specified, got {rho}, {theta}, {zeta}",
        )
        errorif(
            theta is not None,
            NotImplementedError,
            "Constant theta surfaces have not been implemented yet",
        )
        if rho is not None:
            assert (rho >= 0) and (rho <= 1)
            surface = FourierRZToroidalSurface(
                sym=self.sym,
                NFP=self.NFP,
                rho=rho,
            )
            surface.change_resolution(self.M, self.N)

            AR = np.zeros((surface.R_basis.num_modes, self.R_basis.num_modes))
            AZ = np.zeros((surface.Z_basis.num_modes, self.Z_basis.num_modes))

            Js = []
            zernikeR = zernike_radial(
                rho, self.R_basis.modes[:, 0], self.R_basis.modes[:, 1]
            )
            for i, (l, m, n) in enumerate(self.R_basis.modes):
                j = np.argwhere(
                    np.logical_and(
                        surface.R_basis.modes[:, 1] == m,
                        surface.R_basis.modes[:, 2] == n,
                    )
                )
                Js.append(j.flatten())
            Js = np.array(Js)
            # Broadcasting at once is faster. We need to use np.arange to avoid
            # setting the value to the whole row.
            AR[Js[:, 0], np.arange(self.R_basis.num_modes)] = zernikeR

            Js = []
            zernikeZ = zernike_radial(
                rho, self.Z_basis.modes[:, 0], self.Z_basis.modes[:, 1]
            )
            for i, (l, m, n) in enumerate(self.Z_basis.modes):
                j = np.argwhere(
                    np.logical_and(
                        surface.Z_basis.modes[:, 1] == m,
                        surface.Z_basis.modes[:, 2] == n,
                    )
                )
                Js.append(j.flatten())
            Js = np.array(Js)
            # Broadcasting at once is faster. We need to use np.arange to avoid
            # setting the value to the whole row.
            AZ[Js[:, 0], np.arange(self.Z_basis.num_modes)] = zernikeZ

            Rb = AR @ self.R_lmn
            Zb = AZ @ self.Z_lmn
            surface.R_lmn = Rb
            surface.Z_lmn = Zb

            return surface

        if zeta is not None:
            assert (zeta >= 0) and (zeta <= 2 * np.pi)
            surface = ZernikeRZToroidalSection(sym=self.sym, zeta=zeta)
            surface.change_resolution(self.L, self.M)

            AR = np.zeros((surface.R_basis.num_modes, self.R_basis.num_modes))
            AZ = np.zeros((surface.Z_basis.num_modes, self.Z_basis.num_modes))

            for i, (l, m, n) in enumerate(self.R_basis.modes):
                j = np.argwhere(
                    np.logical_and(
                        surface.R_basis.modes[:, 0] == l,
                        surface.R_basis.modes[:, 1] == m,
                    )
                )
                AR[j, i] = fourier(zeta, n, self.NFP)

            for i, (l, m, n) in enumerate(self.Z_basis.modes):
                j = np.argwhere(
                    np.logical_and(
                        surface.Z_basis.modes[:, 0] == l,
                        surface.Z_basis.modes[:, 1] == m,
                    )
                )
                AZ[j, i] = fourier(zeta, n, self.NFP)
            Rb = AR @ self.R_lmn
            Zb = AZ @ self.Z_lmn
            surface.R_lmn = Rb
            surface.Z_lmn = Zb
            return surface

    def get_profile(self, name, grid=None, kind="spline", **kwargs):
        """Return a SplineProfile of the desired quantity.

        Parameters
        ----------
        name : str
            Name of the quantity to compute.
            If list is given, then two names are expected: the quantity to spline
            and its radial derivative.
        grid : Grid, optional
            Grid of coordinates to evaluate at. Defaults to the quadrature grid.
            Note profile will only be a function of the radial coordinate.
        kind : {"power_series", "spline", "fourier_zernike"}
            Type of returned profile.

        Returns
        -------
        profile : SplineProfile
            Radial profile of the desired quantity.

        """
        assert kind in {"power_series", "spline", "fourier_zernike"}
        if grid is None:
            grid = QuadratureGrid(
                self.L_grid,
                self.M_grid,
                self.N_grid,
                self.NFP,
            )
        data = self.compute(name, grid=grid, **kwargs)
        knots = grid.compress(grid.nodes[:, 0])
        if isinstance(name, str):
            f = grid.compress(data[name])
            p = SplineProfile(f, knots, name=name)
        else:
            f, df = map(grid.compress, (data[name[0]], data[name[1]]))
            p = HermiteSplineProfile(f, df, knots, name=name)
        if kind == "power_series":
            p = p.to_powerseries(order=min(self.L, grid.num_rho), xs=knots, sym=True)
        if kind == "fourier_zernike":
            p = p.to_fourierzernike(L=min(self.L, grid.num_rho), xs=knots)
        return p

    def get_axis(self):
        """Return a representation for the magnetic axis.

        Returns
        -------
        axis : FourierRZCurve
            object representing the magnetic axis.
        """
        # value of Zernike polynomials at rho=0 for unique radial modes (+/-1)
        sign_l = np.atleast_2d(((np.arange(0, self.L + 1, 2) / 2) % 2) * -2 + 1).T
        # indices where m=0
        idx0_R = np.where(self.R_basis.modes[:, 1] == 0)[0]
        idx0_Z = np.where(self.Z_basis.modes[:, 1] == 0)[0]
        # indices where l=0 & m=0
        idx00_R = np.where((self.R_basis.modes[:, :2] == [0, 0]).all(axis=1))[0]
        idx00_Z = np.where((self.Z_basis.modes[:, :2] == [0, 0]).all(axis=1))[0]
        # this reshaping assumes the FourierZernike bases are sorted
        R_n = np.sum(
            sign_l * np.reshape(self.R_lmn[idx0_R], (-1, idx00_R.size), order="F"),
            axis=0,
        )
        modes_R = self.R_basis.modes[idx00_R, 2]
        if len(idx00_Z):
            Z_n = np.sum(
                sign_l * np.reshape(self.Z_lmn[idx0_Z], (-1, idx00_Z.size), order="F"),
                axis=0,
            )
            modes_Z = self.Z_basis.modes[idx00_Z, 2]
        else:  # catch cases such as axisymmetry with stellarator symmetry
            Z_n = 0
            modes_Z = 0
        axis = FourierRZCurve(
            R_n,
            Z_n,
            modes_R,
            modes_Z,
            NFP=self.NFP,
            sym=self.sym,
        )
        return axis

    def compute(  # noqa: C901
        self,
        names,
        grid=None,
        params=None,
        transforms=None,
        profiles=None,
        data=None,
        override_grid=True,
        **kwargs,
    ):
        """Compute the quantity given by name on grid.

        If ``grid.coordinates!="rtz"`` then this method may take longer to run
        than usual as a coordinate mapping subproblem will need to be solved.

        Parameters
        ----------
        names : str or array-like of str
            Name(s) of the quantity(s) to compute.
        grid : Grid, optional
            Grid of coordinates to evaluate at. Defaults to the quadrature grid.
        params : dict of ndarray
            Parameters from the equilibrium, such as R_lmn, Z_lmn, i_l, p_l, etc
            Defaults to attributes of self.
        transforms : dict of Transform
            Transforms for R, Z, lambda, etc. Default is to build from grid
        profiles : dict of Profile
            Profile objects for pressure, iota, current, etc. Defaults to attributes
            of self
        data : dict[str, jnp.ndarray]
            Data computed so far, generally output from other compute functions.
            Any vector v = v¹ R̂ + v² ϕ̂ + v³ Ẑ should be given in components
            v = [v¹, v², v³] where R̂, ϕ̂, Ẑ are the normalized basis vectors
            of the cylindrical coordinates R, ϕ, Z.
        override_grid : bool
            If True, override the user supplied grid if necessary and use a full
            resolution grid to compute quantities and then downsample to user requested
            grid. If False, uses only the user specified grid, which may lead to
            inaccurate values for surface or volume averages.

        Returns
        -------
        data : dict of ndarray
            Computed quantity and intermediate variables.

        """
        if isinstance(names, str):
            names = [names]
        if grid is None:
            grid = QuadratureGrid(self.L_grid, self.M_grid, self.N_grid, self.NFP)
        errorif(
            not isinstance(grid, _Grid),
            TypeError,
            msg="must pass in a Grid object for argument grid!"
            f" instead got type {type(grid)}",
        )
        if grid.coordinates != "rtz":
            inbasis = {
                "r": "rho",
                "t": "theta",
                "v": "theta_PEST",
                "a": "alpha",
                "z": "zeta",
                "p": "phi",
            }
            rtz_nodes = self.map_coordinates(
                grid.nodes,
                inbasis=[inbasis[char] for char in grid.coordinates],
                outbasis=("rho", "theta", "zeta"),
                period=grid.period,
            )
            grid = Grid(
                nodes=rtz_nodes,
                coordinates="rtz",
                source_grid=grid,
                sort=False,
                jitable=False,
            )

        method = kwargs.pop("method", "auto")
        if params is None:
            params = get_params(
                names,
                obj=self,
                has_axis=grid.axis.size,
                basis=kwargs.get("basis", "rpz"),
            )
        if profiles is None:
            profiles = get_profiles(
                names, obj=self, grid=grid, basis=kwargs.get("basis", "rpz")
            )
        if transforms is None:
            transforms = get_transforms(
                names,
                obj=self,
                grid=grid,
                method=method,
                **kwargs,
            )
        if data is None:
            data = {}

        p = "desc.equilibrium.equilibrium.Equilibrium"
        deps = set(
            get_data_deps(names, obj=p, has_axis=grid.axis.size, data=data) + names
        )

        def need_src(name):
            # Need to compute these on grid that is paired to the source grid, since
            # the compute logic assume input data is evaluated on those coordinates.
            # We exclude these from the depXdx sets below since the grids we will
            # use to compute those dependencies are coordinate-blind.
            # Example, "fieldline length" has coordinates="r", but requires computing
            # on field line following source grid.
            return bool(data_index[p][name]["source_grid_requirement"])

        # Need to call _grow_seeds so that e.g. "max(fieldline length)" which does not
        # need a source grid to evaluate, still computes "fieldline length"
        # on a grid whose source grid follows field lines.
        # Maybe this can help explain:
        # https://github.com/PlasmaControl/DESC/pull/1024#discussion_r1664918897.
        need_src_deps = _grow_seeds(p, set(filter(need_src, deps)), deps)

        dep0d = {
            dep for dep in deps if is_0d_vol_grid(dep) and dep not in need_src_deps
        }
        # Unless user asks, don't try to recompute stuff which are only dependencies
        # of dep0d. Example, suppose the user supplied grid is a field-line following
        # grid, and the user would like to compute the effective ripple, which requires
        # the scalar R0 as a dependency. The scalar R0 has the following dependencies:
        # R0 <- A <- A(z). Each of these are computable on the quadrature grid, and
        # since R0 is a scalar we can trivially interpolate it back to the user-supplied
        # grid. We don't need to additionally compute A(z) and interpolate it back;
        # it was only needed to compute R0, so we should remove it from the dep1dz list.
        # If we don't remove it from the dep1dz list, then the code would try to create
        # a linear grid with cross-sections at all the unique zeta values in the
        # user-supplied grids. Typically, the user-supplied grid lacks unique_zeta_idx
        # attribute, so this would cause an error.
        dep0d_deps = set(
            get_data_deps(dep0d, obj=p, has_axis=grid.axis.size, data=data)
        )
        # This filter is stronger than the name implies, but the false positives
        # that are filtered out will still get computed with the logic in
        # compute.utils.compute
        # https://github.com/PlasmaControl/DESC/pull/1024#discussion_r1663080423.
        just_dep0d_dep = lambda name: name in dep0d_deps and name not in names
        dep1dr = {
            dep
            for dep in deps
            if is_1dr_rad_grid(dep)
            and not just_dep0d_dep(dep)
            and dep not in need_src_deps
        }
        dep1dz = {
            dep
            for dep in deps
            # By including the additional requirement that dep is not just a dependency
            # of some scalar (0d) quantity, we are ensuring that we do not unnecessarily
            # compute things like A(z) when it was only needed to compute R0, as in the
            # example above.
            if is_1dz_tor_grid(dep)
            and not just_dep0d_dep(dep)
            and dep not in need_src_deps
            # These don't need a special grid, since the transforms are always
            # built on the (rho, theta, zeta) coordinate grid.
            and dep not in ["phi", "zeta"]
        }

        # Whether we need to calculate any dependencies on a special grid.
        calc0d = bool(dep0d)
        calc1dr = bool(dep1dr)
        calc1dz = bool(dep1dz)
        # If the grid samples the full volume, then it is sufficient.
        if grid.L >= self.L_grid and grid.M >= self.M_grid and grid.N >= self.N_grid:
            if isinstance(grid, QuadratureGrid):
                calc0d = calc1dr = calc1dz = False
            if isinstance(grid, LinearGrid):
                calc1dr = calc1dz = False
        else:
            # Warn if best way to compute accurately is increasing resolution.
            for dep in deps:
                req = data_index[p][dep]["resolution_requirement"]
                coords = data_index[p][dep]["coordinates"]
                msg = lambda direction: (
                    f"Dependency {dep} may require more {direction}"
                    " resolution to compute accurately."
                )
                warnif(
                    # if need more radial resolution
                    "r" in req and grid.L < self.L_grid
                    # and won't override grid to one with more radial resolution
                    and not (override_grid and coords in {"z", ""}),
                    ResolutionWarning,
                    msg("radial") + f" got L_grid={grid.L} < {self._L_grid}.",
                )
                warnif(
                    # if need more poloidal resolution
                    "t" in req and grid.M < self.M_grid
                    # and won't override grid to one with more poloidal resolution
                    and not (override_grid and coords in {"r", "z", ""}),
                    ResolutionWarning,
                    msg("poloidal") + f" got M_grid={grid.M} < {self._M_grid}.",
                )
                warnif(
                    # if need more toroidal resolution
                    "z" in req and grid.N < self.N_grid
                    # and won't override grid to one with more toroidal resolution
                    and not (override_grid and coords in {"r", ""}),
                    ResolutionWarning,
                    msg("toroidal") + f" got N_grid={grid.N} < {self._N_grid}.",
                )

        # Now compute dependencies on the proper grids, passing in any available
        # seed data which is already computed and interpolatable.
        # There isn't a single non-repeating order for computing that ensures the
        # dependencies of the dependencies are computed on the proper grid. However,
        # 0d -> 1dr -> 1dz -> rest covers most cases. In cases where it
        # doesn't, the expectation is that developer registers the compute function
        # with a resolution requirement or the user precomputes it.

        if calc0d and override_grid:
            grid0d = QuadratureGrid(self.L_grid, self.M_grid, self.N_grid, self.NFP)
            data0d_seed = {key: data[key] for key in data if is_0d_vol_grid(key)}
            data0d = compute_fun(
                self,
                list(dep0d),
                params=params,
                transforms=get_transforms(
                    dep0d,
                    obj=self,
                    grid=grid0d,
                    method=method,
                    **kwargs,
                ),
                profiles=get_profiles(
                    dep0d, obj=self, grid=grid0d, basis=kwargs.get("basis", "rpz")
                ),
                # If a dependency of something is already computed, use it
                # instead of recomputing it on a potentially bad grid.
                data=data0d_seed,
                **kwargs,
            )
            # These should all be 0d quantities so don't need to compress/expand.
            data0d = {
                key: data0d[key] for key in data0d if key in dep0d and key not in data
            }
            data.update(data0d)

        if (calc1dr or calc1dz) and override_grid:
            data0d_seed = {key: data[key] for key in data if is_0d_vol_grid(key)}
        else:
            data0d_seed = {}
        if calc1dr and override_grid:
            grid1dr = LinearGrid(
                rho=grid.compress(grid.nodes[:, 0], surface_label="rho"),
                M=self.M_grid,
                N=self.N_grid,
                NFP=self.NFP,
                sym=self.sym
                and all(
                    data_index[p][dep]["grid_requirement"].get("sym", True)
                    # TODO (#1206)
                    and not data_index[p][dep]["grid_requirement"].get(
                        "can_fft2", False
                    )
                    for dep in dep1dr
                ),
            )
            data1dr_seed = {
                key: grid1dr.copy_data_from_other(data[key], grid, surface_label="rho")
                for key in data
                if is_1dr_rad_grid(key)
            }
            data1dr = compute_fun(
                self,
                list(dep1dr),
                params=params,
                transforms=get_transforms(
                    dep1dr,
                    obj=self,
                    grid=grid1dr,
                    method=method,
                    **kwargs,
                ),
                profiles=get_profiles(
                    dep1dr, obj=self, grid=grid1dr, basis=kwargs.get("basis", "rpz")
                ),
                # If a dependency of something is already computed, use it
                # instead of recomputing it on a potentially bad grid.
                data=data1dr_seed | data0d_seed,
                **kwargs,
            )
            # Need to make this data broadcast with the data on the original grid.
            data1dr = {
                key: grid.copy_data_from_other(
                    data1dr[key], grid1dr, surface_label="rho"
                )
                for key in data1dr
                if key in dep1dr and key not in data
            }
            data.update(data1dr)

        if calc1dz and override_grid:
            grid1dz = LinearGrid(
                zeta=grid.compress(grid.nodes[:, 2], surface_label="zeta"),
                L=self.L_grid,
                M=self.M_grid,
                NFP=grid.NFP,  # ex: self.NFP>1 but grid.NFP=1 for plot_3d
                sym=self.sym
                and all(
                    data_index[p][dep]["grid_requirement"].get("sym", True)
                    # TODO (#1206)
                    and not data_index[p][dep]["grid_requirement"].get(
                        "can_fft2", False
                    )
                    for dep in dep1dz
                ),
            )
            data1dz_seed = {
                key: grid1dz.copy_data_from_other(data[key], grid, surface_label="zeta")
                for key in data
                if is_1dz_tor_grid(key)
            }
            data1dz = compute_fun(
                self,
                list(dep1dz),
                params=params,
                transforms=get_transforms(
                    dep1dz,
                    obj=self,
                    grid=grid1dz,
                    method=method,
                    **kwargs,
                ),
                profiles=get_profiles(
                    dep1dz, obj=self, grid=grid1dz, basis=kwargs.get("basis", "rpz")
                ),
                # If a dependency of something is already computed, use it
                # instead of recomputing it on a potentially bad grid.
                data=data1dz_seed | data0d_seed,
                **kwargs,
            )
            # Need to make this data broadcast with the data on the original grid.
            data1dz = {
                key: grid.copy_data_from_other(
                    data1dz[key], grid1dz, surface_label="zeta"
                )
                for key in data1dz
                if key in dep1dz and key not in data
            }
            data.update(data1dz)

        data = compute_fun(
            self,
            names,
            params=params,
            transforms=transforms,
            profiles=profiles,
            data=data,
            **kwargs,
        )
        return data

    def map_coordinates(
        self,
        coords,
        inbasis,
        outbasis=("rho", "theta", "zeta"),
        guess=None,
        params=None,
        period=None,
        tol=1e-6,
        maxiter=30,
        full_output=False,
        **kwargs,
    ):
        """Transform coordinates given in ``inbasis`` to ``outbasis``.

        Solves for the computational coordinates that correspond to ``inbasis``,
        then evaluates ``outbasis`` at those locations.

        Performance can often improve significantly given a reasonable initial guess.

        Parameters
        ----------
        coords : ndarray
            Shape (k, 3).
            2D array of input coordinates. Each row is a different point in space.
        inbasis, outbasis : tuple of str
            Labels for input and output coordinates, e.g. ("R", "phi", "Z") or
            ("rho", "alpha", "zeta") or any combination thereof. Labels should be the
            same as the compute function data key.
        guess : jnp.ndarray
            Shape (k, 3).
            Initial guess for the computational coordinates ['rho', 'theta', 'zeta']
            corresponding to ``coords`` in ``inbasis``. If not given, then heuristics
            based on ``inbasis`` or a nearest neighbor search on a grid may be used.
            In general, this must be given to be compatible with JIT.
        params : dict
            Values of equilibrium parameters to use, e.g. ``eq.params_dict``.
        period : tuple of float
            Assumed periodicity for each quantity in ``inbasis``.
            Use ``np.inf`` to denote no periodicity.
        tol : float
            Stopping tolerance.
        maxiter : int
            Maximum number of Newton iterations.
        full_output : bool, optional
            If True, also return a tuple where the first element is the residual from
            the root finding and the second is the number of iterations.
        kwargs : dict, optional
            Additional keyword arguments to pass to ``root`` such as ``maxiter_ls``,
            ``alpha``.

        Returns
        -------
        out : jnp.ndarray
            Shape (k, 3).
            Coordinates mapped from ``inbasis`` to ``outbasis``. Values of NaN will be
            returned for coordinates where root finding did not succeed, possibly
            because the coordinate is not in the plasma volume.
        info : tuple
            2 element tuple containing residuals and number of iterations
            for each point. Only returned if ``full_output`` is True.

        """
        return map_coordinates(
            self,
            coords,
            inbasis,
            outbasis,
            guess,
            params,
            period,
            tol,
            maxiter,
            full_output,
            **kwargs,
        )

    def get_rtz_grid(
        self,
        radial,
        poloidal,
        toroidal,
        coordinates,
        period=(np.inf, np.inf, np.inf),
        jitable=True,
        **kwargs,
    ):
        """Return DESC grid in (rho, theta, zeta) coordinates from given coordinates.

        Create a tensor-product grid from the given coordinates, and return the same
        grid in DESC coordinates.

        Parameters
        ----------
        radial : ndarray
            Sorted unique radial coordinates.
        poloidal : ndarray
            Sorted unique poloidal coordinates.
        toroidal : ndarray
            Sorted unique toroidal coordinates.
        coordinates : str
            Input coordinates that are specified by the arguments, respectively.
            raz : rho, alpha, zeta
            rvp : rho, theta_PEST, phi
            rtz : rho, theta, zeta
        period : tuple of float
            Assumed periodicity of the given coordinates.
            Use ``np.inf`` to denote no periodicity.
        jitable : bool, optional
            If false the returned grid has additional attributes.
            Required to be false to retain nodes at magnetic axis.

        Returns
        -------
        desc_grid : Grid
            DESC coordinate grid for the given coordinates.
        """
        return get_rtz_grid(
            self, radial, poloidal, toroidal, coordinates, period, jitable, **kwargs
        )

    def compute_theta_coords(
        self, flux_coords, L_lmn=None, tol=1e-6, maxiter=20, full_output=False, **kwargs
    ):
        """Find θ (theta_DESC) for given straight field line ϑ (theta_PEST).

        Parameters
        ----------
        flux_coords : ndarray
            Shape (k, 3).
            Straight field line PEST coordinates [ρ, ϑ, ϕ]. Assumes ζ = ϕ.
            Each row is a different point in space.
        L_lmn : ndarray
            Spectral coefficients for lambda. Defaults to ``eq.L_lmn``.
        tol : float
            Stopping tolerance.
        maxiter : int
            Maximum number of Newton iterations.
        full_output : bool, optional
            If True, also return a tuple where the first element is the residual from
            the root finding and the second is the number of iterations.
        kwargs : dict, optional
            Additional keyword arguments to pass to ``root_scalar`` such as
            ``maxiter_ls``, ``alpha``.

        Returns
        -------
        coords : ndarray
            Shape (k, 3).
            DESC computational coordinates [ρ, θ, ζ].
        info : tuple
            2 element tuple containing residuals and number of iterations for each
            point. Only returned if ``full_output`` is True.

        """
        warnif(
            True,
            DeprecationWarning,
            "Use map_coordinates instead of compute_theta_coords.",
        )
        return map_coordinates(
            self,
            coords=flux_coords,
            inbasis=("rho", "theta_PEST", "zeta"),
            outbasis=("rho", "theta", "zeta"),
            params=self.params_dict if L_lmn is None else {"L_lmn": L_lmn},
            tol=tol,
            maxiter=maxiter,
            full_output=full_output,
            **kwargs,
        )

    @execute_on_cpu
    def is_nested(self, grid=None, R_lmn=None, Z_lmn=None, L_lmn=None, msg=None):
        """Check that an equilibrium has properly nested flux surfaces in a plane.

        Does so by checking coordinate Jacobian (sqrt(g)) sign.
        If coordinate Jacobian switches sign somewhere in the volume, this
        indicates that it is zero at some point, meaning surfaces are touching and
        the equilibrium is not nested.

        NOTE: If grid resolution used is too low, or the solution is just barely
        unnested, this function may fail to return the correct answer.

        Parameters
        ----------
        grid  :  Grid, optional
            Grid on which to evaluate the coordinate Jacobian and check for the sign.
            (Default to QuadratureGrid with eq's current grid resolutions)
        R_lmn, Z_lmn, L_lmn : ndarray, optional
            spectral coefficients for R, Z, lambda. Defaults to eq.R_lmn, eq.Z_lmn
        msg : {None, "auto", "manual"}
            Warning to throw if unnested.

        Returns
        -------
        is_nested : bool
            whether the surfaces are nested

        """
        return is_nested(self, grid, R_lmn, Z_lmn, L_lmn, msg)

    def to_sfl(
        self,
        L=None,
        M=None,
        N=None,
        L_grid=None,
        M_grid=None,
        N_grid=None,
        rcond=None,
        copy=False,
    ):
        """Transform this equilibrium to use straight field line coordinates.

        Uses a least squares fit to find FourierZernike coefficients of R, Z, Rb, Zb
        with respect to the straight field line coordinates, rather than the boundary
        coordinates. The new lambda value will be zero.

        NOTE: Though the converted equilibrium will have the same flux surfaces,
        the force balance error will likely be higher than the original equilibrium.

        Parameters
        ----------
        L : int, optional
            radial resolution to use for SFL equilibrium. Default = 1.5*eq.L
        M : int, optional
            poloidal resolution to use for SFL equilibrium. Default = 1.5*eq.M
        N : int, optional
            toroidal resolution to use for SFL equilibrium. Default = 1.5*eq.N
        L_grid : int, optional
            radial spatial resolution to use for fit to new basis. Default = 2*L
        M_grid : int, optional
            poloidal spatial resolution to use for fit to new basis. Default = 2*M
        N_grid : int, optional
            toroidal spatial resolution to use for fit to new basis. Default = 2*N
        rcond : float, optional
            cutoff for small singular values in least squares fit.
        copy : bool, optional
            Whether to update the existing equilibrium or make a copy (Default).

        Returns
        -------
        eq_sfl : Equilibrium
            Equilibrium transformed to a straight field line coordinate representation.

        """
        return to_sfl(self, L, M, N, L_grid, M_grid, N_grid, rcond, copy)

    @property
    def surface(self):
        """Surface: Geometric surface defining boundary conditions."""
        return self._surface

    @surface.setter
    def surface(self, new):
        assert isinstance(
            new, FourierRZToroidalSurface
        ), f"surface should be of type FourierRZToroidalSurface or subclass, got {new}"
        assert (
            self.sym == new.sym
        ), "Surface and Equilibrium must have the same symmetry"
        assert self.NFP == new.NFP, "Surface and Equilibrium must have the same NFP"
        new.change_resolution(self.L, self.M, self.N)
        self._surface = new

    @property
    def axis(self):
        """Curve: object representing the magnetic axis."""
        return self._axis

    @axis.setter
    def axis(self, new):
        assert isinstance(
            new, FourierRZCurve
        ), f"axis should be of type FourierRZCurve or a subclass, got {new}"
        assert self.sym == new.sym, "Axis and Equilibrium must have the same symmetry"
        assert self.NFP == new.NFP, "Axis and Equilibrium must have the same NFP"
        new.change_resolution(self.N)
        self._axis = new

    @property
    def spectral_indexing(self):
        """str: Type of indexing used for the spectral basis."""
        return self._spectral_indexing

    @property
    def sym(self):
        """bool: Whether this equilibrium is stellarator symmetric."""
        return self._sym

    @property
    def bdry_mode(self):
        """str: Method for specifying boundary condition."""
        return self._bdry_mode

    @optimizable_parameter
    @property
    def Psi(self):
        """float: Total toroidal flux within the last closed flux surface in Webers."""
        return self._Psi

    @Psi.setter
    def Psi(self, Psi):
        self._Psi = float(np.squeeze(Psi))

    @property
    def NFP(self):
        """int: Number of (toroidal) field periods."""
        return self._NFP

    @property
    def L(self):
        """int: Maximum radial mode number."""
        return self._L

    @property
    def M(self):
        """int: Maximum poloidal fourier mode number."""
        return self._M

    @property
    def N(self):
        """int: Maximum toroidal fourier mode number."""
        return self._N

    @optimizable_parameter
    @property
    def R_lmn(self):
        """ndarray: Spectral coefficients of R."""
        return self._R_lmn

    @R_lmn.setter
    def R_lmn(self, R_lmn):
        R_lmn = jnp.atleast_1d(jnp.asarray(R_lmn))
        errorif(
            R_lmn.size != self._R_lmn.size,
            ValueError,
            "R_lmn should have the same size as R_basis, "
            + f"got {len(R_lmn)} for basis with {self.R_basis.num_modes} modes",
        )
        self._R_lmn = R_lmn

    @optimizable_parameter
    @property
    def Z_lmn(self):
        """ndarray: Spectral coefficients of Z."""
        return self._Z_lmn

    @Z_lmn.setter
    def Z_lmn(self, Z_lmn):
        Z_lmn = jnp.atleast_1d(jnp.asarray(Z_lmn))
        errorif(
            Z_lmn.size != self._Z_lmn.size,
            ValueError,
            "Z_lmn should have the same size as Z_basis, "
            + f"got {len(Z_lmn)} for basis with {self.Z_basis.num_modes} modes",
        )
        self._Z_lmn = Z_lmn

    @optimizable_parameter
    @property
    def L_lmn(self):
        """ndarray: Spectral coefficients of lambda."""
        return self._L_lmn

    @L_lmn.setter
    def L_lmn(self, L_lmn):
        L_lmn = jnp.atleast_1d(jnp.asarray(L_lmn))
        errorif(
            L_lmn.size != self._L_lmn.size,
            ValueError,
            "L_lmn should have the same size as L_basis, "
            + f"got {len(L_lmn)} for basis with {self.L_basis.num_modes} modes",
        )
        self._L_lmn = L_lmn

    @optimizable_parameter
    @property
    def Rb_lmn(self):
        """ndarray: Spectral coefficients of R at the boundary."""
        return self.surface.R_lmn

    @Rb_lmn.setter
    def Rb_lmn(self, Rb_lmn):
        self.surface.R_lmn = Rb_lmn

    @optimizable_parameter
    @property
    def Zb_lmn(self):
        """ndarray: Spectral coefficients of Z at the boundary."""
        return self.surface.Z_lmn

    @Zb_lmn.setter
    def Zb_lmn(self, Zb_lmn):
        self.surface.Z_lmn = Zb_lmn

    @optimizable_parameter
    @property
    def I(self):  # noqa: E743
        """float: Net toroidal current on the sheet current at the LCFS."""
        return self.surface.I if hasattr(self.surface, "I") else np.empty(0)

    @I.setter
    def I(self, new):  # noqa: E743
        errorif(
            not hasattr(self.surface, "I"),
            ValueError,
            "Attempt to set I on an equilibrium without sheet current",
        )
        self.surface.I = new

    @optimizable_parameter
    @property
    def G(self):
        """float: Net poloidal current on the sheet current at the LCFS."""
        return self.surface.G if hasattr(self.surface, "G") else np.empty(0)

    @G.setter
    def G(self, new):
        errorif(
            not hasattr(self.surface, "G"),
            ValueError,
            "Attempt to set G on an equilibrium without sheet current",
        )
        self.surface.G = new

    @optimizable_parameter
    @property
    def Phi_mn(self):
        """ndarray: coeffs of single-valued part of surface current potential."""
        return self.surface.Phi_mn if hasattr(self.surface, "Phi_mn") else np.empty(0)

    @Phi_mn.setter
    def Phi_mn(self, new):
        errorif(
            not hasattr(self.surface, "Phi_mn"),
            ValueError,
            "Attempt to set Phi_mn on an equilibrium without sheet current",
        )
        self.surface.Phi_mn = new

    @optimizable_parameter
    @property
    def Ra_n(self):
        """ndarray: R coefficients for axis Fourier series."""
        return self.axis.R_n

    @Ra_n.setter
    def Ra_n(self, Ra_n):
        self.axis.R_n = Ra_n

    @optimizable_parameter
    @property
    def Za_n(self):
        """ndarray: Z coefficients for axis Fourier series."""
        return self.axis.Z_n

    @Za_n.setter
    def Za_n(self, Za_n):
        self.axis.Z_n = Za_n

    @property
    def pressure(self):
        """Profile: Pressure (Pa) profile."""
        return self._pressure

    @pressure.setter
    def pressure(self, new):
        self._pressure = parse_profile(new, "pressure")

    @optimizable_parameter
    @property
    def p_l(self):
        """ndarray: Coefficients of pressure profile."""
        return np.empty(0) if self.pressure is None else self.pressure.params

    @p_l.setter
    def p_l(self, p_l):
        errorif(
            self.pressure is None,
            ValueError,
            "Attempt to set pressure on an equilibrium with fixed kinetic profiles",
        )
        self.pressure.params = p_l

    @property
    def anisotropy(self):
        """Profile: Anisotropy profile."""
        return self._anisotropy

    @anisotropy.setter
    def anisotropy(self, new):
        self._anisotropy = parse_profile(new, "anisotropy")

    @optimizable_parameter
    @property
    def a_lmn(self):
        """ndarray: Coefficients of anisotropy profile."""
        return np.empty(0) if self.anisotropy is None else self.anisotropy.params

    @a_lmn.setter
    def a_lmn(self, a_lmn):
        errorif(
            self.anisotropy is None,
            ValueError,
            "Attempt to set anisotropy on an equilibrium without anisotropy profile",
        )
        self.anisotropy.params = a_lmn

    @property
    def electron_temperature(self):
        """Profile: Electron temperature (eV) profile."""
        return self._electron_temperature

    @electron_temperature.setter
    def electron_temperature(self, new):
        self._electron_temperature = parse_profile(new, "electron temperature")

    @optimizable_parameter
    @property
    def Te_l(self):
        """ndarray: Coefficients of electron temperature profile."""
        return (
            np.empty(0)
            if self.electron_temperature is None
            else self.electron_temperature.params
        )

    @Te_l.setter
    def Te_l(self, Te_l):
        errorif(
            self.electron_temperature is None,
            ValueError,
            "Attempt to set electron temperature on an equilibrium with fixed pressure",
        )
        self.electron_temperature.params = Te_l

    @property
    def electron_density(self):
        """Profile: Electron density (m^-3) profile."""
        return self._electron_density

    @electron_density.setter
    def electron_density(self, new):
        self._electron_density = parse_profile(new, "electron density")

    @optimizable_parameter
    @property
    def ne_l(self):
        """ndarray: Coefficients of electron density profile."""
        return (
            np.empty(0)
            if self.electron_density is None
            else self.electron_density.params
        )

    @ne_l.setter
    def ne_l(self, ne_l):
        errorif(
            self.electron_density is None,
            ValueError,
            "Attempt to set electron density on an equilibrium with fixed pressure",
        )
        self.electron_density.params = ne_l

    @property
    def ion_temperature(self):
        """Profile: ion temperature (eV) profile."""
        return self._ion_temperature

    @ion_temperature.setter
    def ion_temperature(self, new):
        self._ion_temperature = parse_profile(new, "ion temperature")

    @optimizable_parameter
    @property
    def Ti_l(self):
        """ndarray: Coefficients of ion temperature profile."""
        return (
            np.empty(0) if self.ion_temperature is None else self.ion_temperature.params
        )

    @Ti_l.setter
    def Ti_l(self, Ti_l):
        errorif(
            self.ion_temperature is None,
            ValueError,
            "Attempt to set ion temperature on an equilibrium with fixed pressure",
        )
        self.ion_temperature.params = Ti_l

    @property
    def atomic_number(self):
        """Profile: Effective atomic number (Z_eff) profile."""
        return self._atomic_number

    @atomic_number.setter
    def atomic_number(self, new):
        self._atomic_number = parse_profile(new, "atomic number")

    @optimizable_parameter
    @property
    def Zeff_l(self):
        """ndarray: Coefficients of effective atomic number profile."""
        return np.empty(0) if self.atomic_number is None else self.atomic_number.params

    @Zeff_l.setter
    def Zeff_l(self, Zeff_l):
        errorif(
            self.atomic_number is None,
            ValueError,
            "Attempt to set atomic number on an equilibrium with fixed pressure",
        )
        self.atomic_number.params = Zeff_l

    @property
    def iota(self):
        """Profile: Rotational transform (iota) profile."""
        return self._iota

    @iota.setter
    def iota(self, new):
        self._iota = parse_profile(new, "iota")
        if self.iota is None:
            return
        warnif(
            self.current is not None,
            UserWarning,
            "Setting rotational transform profile on an equilibrium "
            + "with fixed toroidal current, removing existing toroidal"
            " current profile.",
        )
        self._current = None

    @optimizable_parameter
    @property
    def i_l(self):
        """ndarray: Coefficients of iota profile."""
        return np.empty(0) if self.iota is None else self.iota.params

    @i_l.setter
    def i_l(self, i_l):
        errorif(
            self.iota is None,
            ValueError,
            "Attempt to set parameters of rotational transform on an equilibrium"
            + "with fixed toroidal current",
        )
        self.iota.params = i_l

    @property
    def current(self):
        """Profile: Toroidal current profile (I)."""
        return self._current

    @current.setter
    def current(self, new):
        self._current = parse_profile(new, "current")
        if self.current is None:
            return
        warnif(
            self.iota is not None,
            UserWarning,
            "Setting toroidal current profile on an equilibrium "
            + "with fixed rotational transform, removing existing rotational"
            " transform profile.",
        )
        self._iota = None
        axis_current = np.squeeze(self.current(0.0))
        warnif(
            np.abs(axis_current) > 1e-8,
            UserWarning,
            f"Current on axis is nonzero, got {axis_current:.3e} Amps",
        )

    @optimizable_parameter
    @property
    def c_l(self):
        """ndarray: Coefficients of current profile."""
        return np.empty(0) if self.current is None else self.current.params

    @c_l.setter
    def c_l(self, c_l):
        errorif(
            self.current is None,
            ValueError,
            "Attempt to set parameters of toroidal current on an equilibrium with "
            + "fixed rotational transform",
        )
        self.current.params = c_l
        axis_current = np.squeeze(self.current(0.0))
        warnif(
            np.abs(axis_current) > 1e-8,
            UserWarning,
            f"Current on axis is nonzero, got {axis_current:.3e} Amps",
        )

    @property
    def R_basis(self):
        """FourierZernikeBasis: Spectral basis for R."""
        return self._R_basis

    @property
    def Z_basis(self):
        """FourierZernikeBasis: Spectral basis for Z."""
        return self._Z_basis

    @property
    def L_basis(self):
        """FourierZernikeBasis: Spectral basis for lambda."""
        return self._L_basis

    @property
    def L_grid(self):
        """int: Radial resolution of grid in real space."""
        return self._L_grid

    @L_grid.setter
    def L_grid(self, L_grid):
        if self.L_grid != L_grid:
            self._L_grid = L_grid

    @property
    def M_grid(self):
        """int: Poloidal resolution of grid in real space."""
        return self._M_grid

    @M_grid.setter
    def M_grid(self, M_grid):
        if self.M_grid != M_grid:
            self._M_grid = M_grid

    @property
    def N_grid(self):
        """int: Toroidal resolution of grid in real space."""
        return self._N_grid

    @N_grid.setter
    def N_grid(self, N_grid):
        if self.N_grid != N_grid:
            self._N_grid = N_grid

    @property
    def resolution(self):
        """dict: Spectral and real space resolution parameters of the Equilibrium."""
        return {
            "L": self.L,
            "M": self.M,
            "N": self.N,
            "L_grid": self.L_grid,
            "M_grid": self.M_grid,
            "N_grid": self.N_grid,
        }

    def resolution_summary(self):
        """Print a summary of the spectral and real space resolution."""
        print("Spectral indexing: {}".format(self.spectral_indexing))
        print("Spectral resolution (L,M,N)=({},{},{})".format(self.L, self.M, self.N))
        print(
            "Node resolution (L,M,N)=({},{},{})".format(
                self.L_grid, self.M_grid, self.N_grid
            )
        )

    @classmethod
    def from_near_axis(
        cls,
        na_eq,
        r=0.1,
        L=None,
        M=8,
        N=None,
        ntheta=None,
        spectral_indexing="ansi",
        w=2,
    ):
        """Initialize an Equilibrium from a near-axis solution.

        Parameters
        ----------
        na_eq : Qsc or Qic
            Near-axis solution generated by pyQSC or pyQIC.
        r : float
            Radius of the desired boundary surface (in meters).
        L : int (optional)
            Radial resolution. Default 2*M for ``spectral_indexing=='fringe'``, else M
        M : int (optional)
            Poloidal resolution. Default is 8
        N : int (optional)
            Toroidal resolution. Default is M.
            If N=np.inf, the max resolution provided by na_eq.nphi is used.
        ntheta : int, optional
            Number of poloidal grid points used in the conversion. Default 2*M+1
        spectral_indexing : str (optional)
            Type of Zernike indexing scheme to use. Default ``'ansi'``
        w : float
            Weight exponent for fit. Surfaces are fit using a weighted least squares
            using weight = 1/rho**w, so that points near axis are captured more
            accurately.

        Returns
        -------
        eq : Equilibrium
            Equilibrium approximation of the near-axis solution.

        """
        try:
            # default resolution parameters
            if L is None:
                if spectral_indexing == "ansi":
                    L = M
                elif spectral_indexing == "fringe":
                    L = 2 * M
            if N is None:
                N = M
            if N == np.inf:
                N = int((na_eq.nphi - 1) / 2)

            if ntheta is None:
                ntheta = 2 * M + 1

            inputs = {
                "Psi": np.pi * r**2 * na_eq.Bbar,
                "NFP": na_eq.nfp,
                "L": L,
                "M": M,
                "N": N,
                "sym": not na_eq.lasym,
                "spectral_indexing": spectral_indexing,
                "pressure": np.array([[0, -na_eq.p2 * r**2], [2, na_eq.p2 * r**2]]),
                "iota": None,
                "current": np.array([[2, 2 * np.pi / mu_0 * na_eq.I2 * r**2]]),
                "axis": FourierRZCurve(
                    R_n=np.concatenate((np.flipud(na_eq.rs[1:]), na_eq.rc)),
                    Z_n=np.concatenate((np.flipud(na_eq.zs[1:]), na_eq.zc)),
                    NFP=na_eq.nfp,
                ),
                "surface": None,
            }
        except AttributeError as e:
            raise ValueError("Input must be a pyQSC or pyQIC solution.") from e

        rho, _ = special.js_roots(L, 2, 2)
        grid = LinearGrid(rho=rho, theta=ntheta, zeta=na_eq.phi, NFP=na_eq.nfp)
        basis_R = FourierZernikeBasis(
            L=L,
            M=M,
            N=N,
            NFP=na_eq.nfp,
            sym="cos" if not na_eq.lasym else False,
            spectral_indexing=spectral_indexing,
        )
        basis_Z = FourierZernikeBasis(
            L=L,
            M=M,
            N=N,
            NFP=na_eq.nfp,
            sym="sin" if not na_eq.lasym else False,
            spectral_indexing=spectral_indexing,
        )
        basis_L = FourierZernikeBasis(
            L=L,
            M=M,
            N=N,
            NFP=na_eq.nfp,
            sym="sin" if not na_eq.lasym else False,
            spectral_indexing=spectral_indexing,
        )

        transform_R = Transform(grid, basis_R, method="direct1")
        transform_Z = Transform(grid, basis_Z, method="direct1")
        transform_L = Transform(grid, basis_L, method="direct1")
        A_R = transform_R.matrices["direct1"][0][0][0]
        A_Z = transform_Z.matrices["direct1"][0][0][0]
        A_L = transform_L.matrices["direct1"][0][0][0]

        W = 1 / grid.nodes[:, 0].flatten() ** w
        A_Rw = A_R * W[:, None]
        A_Zw = A_Z * W[:, None]
        A_Lw = A_L * W[:, None]

        R_1D = np.zeros((grid.num_nodes,))
        Z_1D = np.zeros((grid.num_nodes,))
        L_1D = np.zeros((grid.num_nodes,))
        for rho_i in rho:
            R_2D, Z_2D, phi0_2D = na_eq.Frenet_to_cylindrical(r * rho_i, ntheta)
            phi_cyl_ax = np.linspace(
                0, 2 * np.pi / na_eq.nfp, na_eq.nphi, endpoint=False
            )
            nu_B_ax = na_eq.nu_spline(phi_cyl_ax)
            phi_B = phi_cyl_ax + nu_B_ax
            nu_B = phi_B - phi0_2D
            idx = np.nonzero(grid.nodes[:, 0] == rho_i)[0]
            R_1D[idx] = R_2D.flatten(order="F")
            Z_1D[idx] = Z_2D.flatten(order="F")
            L_1D[idx] = -nu_B.flatten(order="F") * na_eq.iota

        inputs["R_lmn"] = np.linalg.lstsq(A_Rw, R_1D * W, rcond=None)[0]
        inputs["Z_lmn"] = np.linalg.lstsq(A_Zw, Z_1D * W, rcond=None)[0]
        inputs["L_lmn"] = np.linalg.lstsq(A_Lw, L_1D * W, rcond=None)[0]

        eq = Equilibrium(**inputs)
        eq.surface = eq.get_surface_at(rho=1)

        return eq

    @classmethod
    def from_input_file(cls, path, **kwargs):
        """Create an Equilibrium from information in a DESC or VMEC input file.

        Parameters
        ----------
        path : Path-like or str
            Path to DESC or VMEC input file.
        **kwargs : dict, optional
            keyword arguments to pass to the constructor of the
            Equilibrium being created.

        Returns
        -------
        Equilibrium : Equilibrium
            Equilibrium generated from the given input file.

        """
        inputs = InputReader().parse_inputs(path)[-1]
        if (inputs["bdry_ratio"] is not None) and (inputs["bdry_ratio"] != 1):
            warnings.warn(
                "`bdry_ratio` is intended as an input for the continuation method."
                "`bdry_ratio`=1 uses the given surface modes as is, any other scalar "
                "value will scale the non-axisymmetric  modes by that value. The "
                "final value of `bdry_ratio` in the input file is "
                f"{inputs['bdry_ratio']}, this means the created Equilibrium won't "
                "have the given surface but a scaled version instead."
            )
        inputs["surface"][:, 1:3] = inputs["surface"][:, 1:3].astype(int)
        # remove the keys (pertaining to continuation and solver tols)
        # that an Equilibrium does not need
        unused_keys = [
            "pres_ratio",
            "bdry_ratio",
            "pert_order",
            "ftol",
            "xtol",
            "gtol",
            "maxiter",
            "objective",
            "optimizer",
            "bdry_mode",
            "output_path",
            "verbose",
        ]
        [inputs.pop(key) for key in unused_keys]
        inputs.update(kwargs)
        return cls(**inputs)

    def solve(
        self,
        objective="force",
        constraints=None,
        optimizer="lsq-exact",
        ftol=None,
        xtol=None,
        gtol=None,
        maxiter=None,
        x_scale="auto",
        options=None,
        verbose=1,
        copy=False,
    ):
        """Solve to find the equilibrium configuration.

        Parameters
        ----------
        objective : {"force", "forces", "energy"}
            Objective function to solve. Default = force balance on unified grid.
        constraints : Tuple
            set of constraints to enforce. Default = fixed boundary/profiles
        optimizer : str or Optimizer (optional)
            optimizer to use
        ftol, xtol, gtol : float
            stopping tolerances. `None` will use defaults for given optimizer.
        maxiter : int
            Maximum number of solver steps.
        x_scale : array_like or ``'auto'``, optional
            Characteristic scale of each variable. Setting ``x_scale`` is equivalent
            to reformulating the problem in scaled variables ``xs = x / x_scale``.
            An alternative view is that the size of a trust region along jth
            dimension is proportional to ``x_scale[j]``. Improved convergence may
            be achieved by setting ``x_scale`` such that a step of a given size
            along any of the scaled variables has a similar effect on the cost
            function. If set to ``'auto'``, the scale is iteratively updated using the
            inverse norms of the columns of the Jacobian or Hessian matrix.
        options : dict
            Dictionary of additional options to pass to optimizer.
        verbose : int
            Level of output.
        copy : bool
            Whether to return the current equilibrium or a copy (leaving the original
            unchanged).

        Returns
        -------
        eq : Equilibrium
            Either this equilibrium or a copy, depending on "copy" argument.
        result : OptimizeResult
            The optimization result represented as a ``OptimizeResult`` object.
            Important attributes are: ``x`` the solution array, ``success`` a
            Boolean flag indicating if the optimizer exited successfully and
            ``message`` which describes the cause of the termination. See
            `OptimizeResult` for a description of other attributes.

        """
        if constraints is None:
            constraints = get_fixed_boundary_constraints(eq=self)
        if not isinstance(objective, ObjectiveFunction):
            objective = get_equilibrium_objective(eq=self, mode=objective)
        if not isinstance(optimizer, Optimizer):
            optimizer = Optimizer(optimizer)
        if not isinstance(constraints, (list, tuple)):
            constraints = tuple([constraints])

        warnif(
            self.N > self.N_grid or self.M > self.M_grid or self.L > self.L_grid,
            msg="Equilibrium has one or more spectral resolutions "
            + "greater than the corresponding collocation grid resolution! "
            + "This is not recommended and may result in poor convergence. "
            + "Set grid resolutions to be higher, (i.e. eq.N_grid=2*eq.N) "
            + "to avoid this warning.",
        )
        errorif(
            self.bdry_mode == "poincare",
            NotImplementedError,
            "Solving equilibrium with poincare XS as BC is not supported yet "
            + "on master branch.",
        )

        things, result = optimizer.optimize(
            self,
            objective,
            constraints,
            ftol=ftol,
            xtol=xtol,
            gtol=gtol,
            x_scale=x_scale,
            verbose=verbose,
            maxiter=maxiter,
            options=options,
            copy=copy,
        )

        return things[0], result

    def optimize(
        self,
        objective=None,
        constraints=None,
        optimizer="proximal-lsq-exact",
        ftol=None,
        xtol=None,
        gtol=None,
        ctol=None,
        maxiter=None,
        x_scale="auto",
        options=None,
        verbose=1,
        copy=False,
    ):
        """Optimize an equilibrium for an objective.

        Parameters
        ----------
        objective : ObjectiveFunction
            Objective function to optimize.
        constraints : Objective or tuple of Objective
            Objective function to satisfy. Default = fixed-boundary force balance.
        optimizer : str or Optimizer (optional)
            optimizer to use
        ftol, xtol, gtol, ctol : float
            stopping tolerances. `None` will use defaults for given optimizer.
        maxiter : int
            Maximum number of solver steps.
        x_scale : array_like or ``'auto'``, optional
            Characteristic scale of each variable. Setting ``x_scale`` is equivalent
            to reformulating the problem in scaled variables ``xs = x / x_scale``.
            An alternative view is that the size of a trust region along jth
            dimension is proportional to ``x_scale[j]``. Improved convergence may
            be achieved by setting ``x_scale`` such that a step of a given size
            along any of the scaled variables has a similar effect on the cost
            function. If set to ``'auto'``, the scale is iteratively updated using the
            inverse norms of the columns of the Jacobian or Hessian matrix.
        options : dict
            Dictionary of additional options to pass to optimizer.
        verbose : int
            Level of output.
        copy : bool
            Whether to return the current equilibrium or a copy (leaving the original
            unchanged).

        Returns
        -------
        eq : Equilibrium
            Either this equilibrium or a copy, depending on "copy" argument.
        result : OptimizeResult
            The optimization result represented as a ``OptimizeResult`` object.
            Important attributes are: ``x`` the solution array, ``success`` a
            Boolean flag indicating if the optimizer exited successfully and
            ``message`` which describes the cause of the termination. See
            `OptimizeResult` for a description of other attributes.

        """
        if not isinstance(optimizer, Optimizer):
            optimizer = Optimizer(optimizer)
        if constraints is None:
            constraints = get_fixed_boundary_constraints(eq=self)
            constraints = (ForceBalance(eq=self), *constraints)
        if not isinstance(constraints, (list, tuple)):
            constraints = tuple([constraints])

        things, result = optimizer.optimize(
            self,
            objective,
            constraints,
            ftol=ftol,
            xtol=xtol,
            gtol=gtol,
            ctol=ctol,
            x_scale=x_scale,
            verbose=verbose,
            maxiter=maxiter,
            options=options,
            copy=copy,
        )

        return things[0], result

    def perturb(
        self,
        deltas,
        objective=None,
        constraints=None,
        order=2,
        tr_ratio=0.1,
        weight="auto",
        include_f=True,
        verbose=1,
        copy=False,
    ):
        """Perturb an equilibrium.

        Parameters
        ----------
        objective : ObjectiveFunction
            Objective function to satisfy. Default = force balance.
        constraints : Objective or tuple of Objective
            Constraint function to satisfy. Default = fixed-boundary.
        deltas : dict of ndarray
            Deltas for perturbations. Keys should names of Equilibrium attributes
            ("p_l",  "Rb_lmn", "L_lmn" etc.) and values of arrays of desired change in
            the attribute.
        order : {0,1,2,3}
            Order of perturbation (0=none, 1=linear, 2=quadratic, etc.)
        tr_ratio : float or array of float
            Radius of the trust region, as a fraction of ||x||.
            Enforces ||dx1|| <= tr_ratio*||x|| and ||dx2|| <= tr_ratio*||dx1||.
            If a scalar, uses the same ratio for all steps. If an array, uses the first
            element for the first step and so on.
        weight : ndarray, "auto", or None, optional
            1d or 2d array for weighted least squares. 1d arrays are turned into
            diagonal matrices. Default is to weight by (mode number)**2. None applies
            no weighting.
        include_f : bool, optional
            Whether to include the 0th order objective residual in the perturbation
            equation. Including this term can improve force balance if the perturbation
            step is large, but can result in too large a step if the perturbation
            is small.
        verbose : int
            Level of output.
        copy : bool, optional
            Whether to update the existing equilibrium or make a copy (Default).

        Returns
        -------
        eq_new : Equilibrium
            Perturbed equilibrium.

        """
        if objective is None:
            objective = get_equilibrium_objective(eq=self)
        if constraints is None:
            if "Ra_n" in deltas or "Za_n" in deltas:
                constraints = get_fixed_axis_constraints(eq=self)
            else:
                constraints = get_fixed_boundary_constraints(eq=self)

        eq = perturb(
            self,
            objective,
            constraints,
            deltas,
            order=order,
            tr_ratio=tr_ratio,
            weight=weight,
            include_f=include_f,
            verbose=verbose,
            copy=copy,
        )

        return eq


class EquilibriaFamily(IOAble, MutableSequence):
    """EquilibriaFamily stores a list of Equilibria.

    Has methods for solving complex equilibria using a multi-grid continuation method.

    Parameters
    ----------
    args : Equilibrium, dict or list of dict
        Should be either:

        * An Equilibrium (or several)
        * A dictionary of inputs (or several) to create a equilibria
        * A single list of dictionaries, one for each equilibrium in a continuation.
        * Nothing, to create an empty family.

        For more information see inputs required by ``'Equilibrium'``.
    """

    _io_attrs_ = ["_equilibria"]

    def __init__(self, *args):
        # we use ensure_nested=False here for all but the first iteration
        # because it is assumed the family
        # will be solved with a continuation method, so there's no need for the
        # fancy coordinate mapping stuff since it will just be overwritten during
        # solve_continuation
        self.equilibria = []
        if len(args) == 1 and isinstance(args[0], list):
            for i, inp in enumerate(args[0]):
                # ensure that first step is nested
                ensure_nested_bool = True if i == 0 else False
                self.equilibria.append(
                    Equilibrium(
                        **inp, ensure_nested=ensure_nested_bool, check_kwargs=False
                    )
                )
        else:
            for i, arg in enumerate(args):
                if isinstance(arg, Equilibrium):
                    self.equilibria.append(arg)
                elif isinstance(arg, dict):
                    ensure_nested_bool = True if i == 0 else False
                    self.equilibria.append(
                        Equilibrium(
                            **arg, ensure_nested=ensure_nested_bool, check_kwargs=False
                        )
                    )
                else:
                    raise TypeError(
                        "Args to create EquilibriaFamily should either be "
                        + "Equilibrium or dictionary"
                    )

    def solve_continuation(
        self,
        objective="force",
        optimizer="lsq-exact",
        pert_order=2,
        ftol=None,
        xtol=None,
        gtol=None,
        maxiter=100,
        verbose=1,
        checkpoint_path=None,
    ):
        """Solve for an equilibrium by continuation method.

        Steps through an EquilibriaFamily, solving each equilibrium, and uses
        perturbations to step between different profiles/boundaries.

        Uses the previous step as an initial guess for each solution.

        Parameters
        ----------
        eqfam : EquilibriaFamily or list of Equilibria
            Equilibria to solve for at each step.
        objective : str or ObjectiveFunction (optional)
            function to solve for equilibrium solution
        optimizer : str or Optimizer (optional)
            optimizer to use
        pert_order : int or array of int
            order of perturbations to use. If array-like, should be same length as
            family to specify different values for each step.
        ftol, xtol, gtol : float or array-like of float
            stopping tolerances for subproblem at each step. `None` will use defaults
            for given optimizer.
        maxiter : int or array-like of int
            maximum number of iterations in each equilibrium subproblem.
        verbose : integer
            * 0: no output
            * 1: summary of each iteration
            * 2: as above plus timing information
            * 3: as above plus detailed solver output
        checkpoint_path : str or path-like
            file to save checkpoint data (Default value = None)

        Returns
        -------
        eqfam : EquilibriaFamily
            family of equilibria for the intermediate steps, where the last member is
            the final desired configuration,

        """
        from desc.continuation import solve_continuation

        return solve_continuation(
            self,
            objective,
            optimizer,
            pert_order,
            ftol,
            xtol,
            gtol,
            maxiter,
            verbose,
            checkpoint_path,
        )

    @classmethod
    def solve_continuation_automatic(
        cls,
        eq,
        objective="force",
        optimizer="lsq-exact",
        pert_order=2,
        ftol=None,
        xtol=None,
        gtol=None,
        maxiter=100,
        verbose=1,
        checkpoint_path=None,
        **kwargs,
    ):
        """Solve for an equilibrium using an automatic continuation method.

        By default, the method first solves for a no pressure tokamak, then a finite
        beta tokamak, then a finite beta stellarator. Currently hard coded to take a
        fixed number of perturbation steps based on conservative estimates and testing.
        In the future, continuation stepping will be done adaptively.

        Parameters
        ----------
        eq : Equilibrium
            Unsolved Equilibrium with the final desired boundary, profiles, resolution.
        objective : str or ObjectiveFunction (optional)
            function to solve for equilibrium solution
        optimizer : str or Optimizer (optional)
            optimizer to use
        pert_order : int
            order of perturbations to use.
        ftol, xtol, gtol : float
            stopping tolerances for subproblem at each step. `None` will use defaults
            for given optimizer.
        maxiter : int
            maximum number of iterations in each equilibrium subproblem.
        verbose : integer
            * 0: no output
            * 1: summary of each iteration
            * 2: as above plus timing information
            * 3: as above plus detailed solver output
        checkpoint_path : str or path-like
            file to save checkpoint data (Default value = None)
        **kwargs : dict, optional
            * ``mres_step``: int, default 6. The amount to increase Mpol by at each
              continuation step
            * ``pres_step``: float, ``0<=pres_step<=1``, default 0.5. The amount to
              increase pres_ratio by at each continuation step
            * ``bdry_step``: float, ``0<=bdry_step<=1``, default 0.25. The amount to
              increase bdry_ratio by at each continuation step

        Returns
        -------
        eqfam : EquilibriaFamily
            family of equilibria for the intermediate steps, where the last member is
            the final desired configuration,

        """
        from desc.continuation import solve_continuation_automatic

        return solve_continuation_automatic(
            eq,
            objective,
            optimizer,
            pert_order,
            ftol,
            xtol,
            gtol,
            maxiter,
            verbose,
            checkpoint_path,
            **kwargs,
        )

    @property
    def equilibria(self):
        """list: Equilibria contained in the family."""
        return self._equilibria

    @equilibria.setter
    def equilibria(self, equil):
        if isinstance(equil, tuple):
            equil = list(equil)
        elif isinstance(equil, np.ndarray):
            equil = equil.tolist()
        elif not isinstance(equil, list):
            equil = [equil]
        if len(equil) and not all([isinstance(eq, Equilibrium) for eq in equil]):
            raise ValueError(
                "Members of EquilibriaFamily should be of type Equilibrium or subclass."
            )
        self._equilibria = list(equil)

    def __getitem__(self, i):
        return self._equilibria[i]

    def __setitem__(self, i, new_item):
        if not isinstance(new_item, Equilibrium):
            raise ValueError(
                "Members of EquilibriaFamily should be of type Equilibrium or subclass."
            )
        self._equilibria[i] = new_item

    def __delitem__(self, i):
        del self._equilibria[i]

    def __len__(self):
        return len(self._equilibria)

    def insert(self, i, new_item):
        """Insert a new Equilibrium into the family at position i."""
        if not isinstance(new_item, Equilibrium):
            raise ValueError(
                "Members of EquilibriaFamily should be of type Equilibrium or subclass."
            )
        self._equilibria.insert(i, new_item)
