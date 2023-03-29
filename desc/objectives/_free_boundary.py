"""Objectives for solving free boundary equilibria."""

from scipy.constants import mu_0

from desc.backend import jnp
from desc.compute import compute as compute_fun
from desc.compute import get_params, get_profiles, get_transforms
from desc.grid import LinearGrid
from desc.nestor import Nestor
from desc.objectives.objective_funs import _Objective
from desc.utils import Timer

from .normalization import compute_scaling_factors


class BoundaryErrorNESTOR(_Objective):
    """Pressure balance across LCFS.

    Uses NESTOR algorithm to compute B_vac such that (B_vac + B_coil)*n=0,
    then calculates the pressure mismatch across the boundary:

        1/2mu0*(B_vac + B_coil)^2 - 1/2mu0*B_plasma^2 - p


    Parameters
    ----------
    ext_field : MagneticField
        External field produced by coils.
    eq : Equilibrium, optional
        Equilibrium that will be optimized to satisfy the Objective.
    target : float, ndarray, optional
        Target value(s) of the objective. Only used if bounds is None.
        len(target) must be equal to Objective.dim_f
    bounds : tuple, optional
        Lower and upper bounds on the objective. Overrides target.
        len(bounds[0]) and len(bounds[1]) must be equal to Objective.dim_f
    weight : float, ndarray, optional
        Weighting to apply to the Objective, relative to other Objectives.
        len(weight) must be equal to Objective.dim_f
    mf, nf : integer
        maximum poloidal and toroidal mode numbers to use for NESTOR scalar potential.
    ntheta, nzeta : int
        number of grid points in poloidal, toroidal directions to use in NESTOR.
    normalize : bool
        Whether to compute the error in physical units or non-dimensionalize.
    normalize_target : bool
        Whether target and bounds should be normalized before comparing to computed
        values. If `normalize` is `True` and the target is in physical units,
        this should also be set to True.
    grid : Grid, ndarray, optional
        Collocation grid containing the nodes to evaluate at.
    name : str
        Name of the objective function.

    """

    _scalar = False
    _linear = False
    _print_value_fmt = "Boundary Force Error: {:10.3e} "
    _units = "(N)"

    def __init__(
        self,
        ext_field,
        eq=None,
        target=0,
        bounds=None,
        weight=1,
        mf=None,
        nf=None,
        ntheta=None,
        nzeta=None,
        normalize=True,
        normalize_target=True,
        name="NESTOR Boundary",
    ):

        self.mf = mf
        self.nf = nf
        self.ntheta = ntheta
        self.nzeta = nzeta
        self.ext_field = ext_field
        super().__init__(
            eq=eq,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            name=name,
        )

    def build(self, eq, use_jit=True, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        self.mf = eq.M + 1 if self.mf is None else self.mf
        self.nf = eq.N if self.nf is None else self.nf
        self.ntheta = 4 * eq.M + 1 if self.ntheta is None else self.ntheta
        self.nzeta = 4 * eq.N + 1 if self.nzeta is None else self.nzeta

        eq._sym = False
        self.nest = Nestor(
            eq, self.ext_field, self.mf, self.nf, self.ntheta, self.nzeta
        )
        eq._sym = True
        self.grid = LinearGrid(rho=1, theta=self.ntheta, zeta=self.nzeta, NFP=eq.NFP)
        self._data_keys = ["current", "|B|^2", "p", "|e_theta x e_zeta|"]
        self._args = get_params(self._data_keys)

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._profiles = get_profiles(self._data_keys, eq=eq, grid=self.grid)
        self._transforms = get_transforms(self._data_keys, eq=eq, grid=self.grid)

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        self._dim_f = self.grid.num_nodes

        if self._normalize:
            scales = compute_scaling_factors(eq)
            # local quantity, want to divide by number of nodes
            self._normalization = scales["p"] / jnp.sqrt(self._dim_f)

        super().build(eq=eq, use_jit=use_jit, verbose=verbose)

    def compute(self, *args, **kwargs):
        """Compute boundary force error.

        Parameters
        ----------
        R_lmn : ndarray
            Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate (m).
        Z_lmn : ndarray
            Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordinate (m).
        L_lmn : ndarray
            Spectral coefficients of lambda(rho,theta,zeta) -- poloidal stream function.
        p_l : ndarray
            Spectral coefficients of p(rho) -- pressure profile (Pa).
        i_l : ndarray
            Spectral coefficients of iota(rho) -- rotational transform profile.
        c_l : ndarray
            Spectral coefficients of I(rho) -- toroidal current profile (A).
        Psi : float
            Total toroidal magnetic flux within the last closed flux surface (Wb).
        Te_l : ndarray
            Spectral coefficients of Te(rho) -- electron temperature profile (eV).
        ne_l : ndarray
            Spectral coefficients of ne(rho) -- electron density profile (1/m^3).
        Ti_l : ndarray
            Spectral coefficients of Ti(rho) -- ion temperature profile (eV).
        Zeff_l : ndarray
            Spectral coefficients of Zeff(rho) -- effective atomic number profile.

        Returns
        -------
        f : ndarray
            Boundary force error (N).

        """
        params = self._parse_args(*args, **kwargs)
        data = compute_fun(
            self._data_keys,
            params=params,
            transforms=self._transforms,
            profiles=self._profiles,
        )

        ctor = jnp.mean(data["current"])
        out = self.nest.compute(params["R_lmn"], params["Z_lmn"], ctor)
        grid = self.nest._Rb_transform.grid
        bsq = out[1]["|B|^2"].reshape((grid.num_zeta, grid.num_theta)).T.flatten()
        bv = bsq / (2 * mu_0)

        bp = data["|B|^2"] / (2 * mu_0)
        w = self.grid.weights
        g = data["|e_theta x e_zeta|"]
        return (bv - bp - data["p"]) * w * g
