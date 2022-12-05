"""Generic objectives that don't belong anywhere else."""

import desc.compute as compute_funs
from desc.compute import compute_boozer_magnetic_field, data_index
from desc.compute.utils import compress, get_params, get_profiles, get_transforms
from desc.grid import LinearGrid, QuadratureGrid
from desc.utils import Timer

from .objective_funs import _Objective


class GenericObjective(_Objective):
    """A generic objective that can compute any quantity from the `data_index`.

    Parameters
    ----------
    f : str
        Name of the quantity to compute.
    eq : Equilibrium, optional
        Equilibrium that will be optimized to satisfy the Objective.
    target : float, ndarray, optional
        Target value(s) of the objective.
        len(target) must be equal to Objective.dim_f
    weight : float, ndarray, optional
        Weighting to apply to the Objective, relative to other Objectives.
        len(weight) must be equal to Objective.dim_f
    grid : Grid, ndarray, optional
        Collocation grid containing the nodes to evaluate at.
    name : str
        Name of the objective function.

    """

    _scalar = False
    _linear = False

    def __init__(self, f, eq=None, target=0, weight=1, grid=None, name="generic"):

        self.f = f
        self.grid = grid
        super().__init__(eq=eq, target=target, weight=weight, name=name)
        self._print_value_fmt = (
            "Residual: {:10.3e} (" + data_index[self.f]["units"] + ")"
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
        if self.grid is None:
            self.grid = QuadratureGrid(eq.L_grid, eq.M_grid, eq.N_grid, eq.NFP)

        self._dim_f = self.grid.num_nodes
        self.fun = getattr(compute_funs, data_index[self.f]["fun"])
        self._args = get_params(self.f)
        self._profiles = get_profiles(self.f, eq=eq, grid=self.grid)
        self._transforms = get_transforms(self.f, eq=eq, grid=self.grid)
        super().build(eq=eq, use_jit=use_jit, verbose=verbose)

    def compute(self, **params):
        """Compute the quantity.

        Parameters
        ----------
        args : list of ndarray
            Any of the arguments given in `arg_order`.

        Returns
        -------
        f : ndarray
            Computed quantity.

        """
        data = self.fun(params, self._transforms, self._profiles)
        f = data[self.f]
        return self._shift_scale(f)


# TODO: move this class to a different file (not generic)
class ToroidalCurrent(_Objective):
    """Toroidal current enclosed by a surface.

    Parameters
    ----------
    eq : Equilibrium, optional
        Equilibrium that will be optimized to satisfy the Objective.
    target : float, ndarray, optional
        Target value(s) of the objective.
        len(target) must be equal to Objective.dim_f
    weight : float, ndarray, optional
        Weighting to apply to the Objective, relative to other Objectives.
        len(weight) must be equal to Objective.dim_f
    grid : Grid, ndarray, optional
        Collocation grid containing the nodes to evaluate at.
    name : str
        Name of the objective function.

    """

    _scalar = True
    _linear = False

    def __init__(self, eq=None, target=0, weight=1, grid=None, name="toroidal current"):

        self.grid = grid
        super().__init__(eq=eq, target=target, weight=weight, name=name)
        self._print_value_fmt = "Toroidal current: {:10.3e} (A)"

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
        if self.grid is None:
            self.grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym)

        self._dim_f = self.grid.num_rho
        self._data_keys = ["current"]

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._profiles = get_profiles(*self._data_keys, eq=eq, grid=self.grid)
        self._transforms = get_transforms(*self._data_keys, eq=eq, grid=self.grid)

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        super().build(eq=eq, use_jit=use_jit, verbose=verbose)

    def compute(self, R_lmn, Z_lmn, L_lmn, i_l, c_l, Psi, **kwargs):
        """Compute toroidal current.

        Parameters
        ----------
        R_lmn : ndarray
            Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate (m).
        Z_lmn : ndarray
            Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordinate (m).
        L_lmn : ndarray
            Spectral coefficients of lambda(rho,theta,zeta) -- poloidal stream function.
        i_l : ndarray
            Spectral coefficients of iota(rho) -- rotational transform profile.
        c_l : ndarray
            Spectral coefficients of I(rho) -- toroidal current profile.
        Psi : float
            Total toroidal magnetic flux within the last closed flux surface (Wb).

        Returns
        -------
        I : float
            Toroidal current (A).

        """
        params = {
            "R_lmn": R_lmn,
            "Z_lmn": Z_lmn,
            "L_lmn": L_lmn,
            "i_l": i_l,
            "c_l": c_l,
            "Psi": Psi,
        }
        data = compute_boozer_magnetic_field(
            params,
            self._transforms,
            self._profiles,
        )
        I = compress(self.grid, data["current"], surface_label="rho")
        return self._shift_scale(I)
