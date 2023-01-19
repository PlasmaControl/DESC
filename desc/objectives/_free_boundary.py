"""Objectives for solving free boundary equilibria."""

import numpy as np
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
    _scalar = False
    _linear = False
    _print_value_fmt = "Boundary Pressure Imbalance: {:10.3e} "
    _units = "Pa"

    def __init__(
        self,
        ext_field,
        eq=None,
        target=0,
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
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            name=name,
        )

    def build(self, eq, use_jit=True, verbose=1):

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


        params = self._parse_args(*args, **kwargs)
        data = compute_fun(
            self._data_keys,
            params=params,
            transforms=self._transforms,
            profiles=self._profiles,
        )

        ctor = jnp.mean(data['current'])
        out = self.nest.compute(params['R_lmn'], params['Z_lmn'], ctor)
        grid = self.nest._Rb_transform.grid
        bsq = out[1]["|B|^2"].reshape((grid.num_zeta, grid.num_theta)).T.flatten()
        bv = bsq / (2 * mu_0)

        bp = data['|B|^2'] / (2 * mu_0)
        w = self.grid.weights
        g = data["|e_theta x e_zeta|"]
        return (bv - bp - data['p']) * w * g
