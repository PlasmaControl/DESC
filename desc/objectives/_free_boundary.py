"""Objectives for solving free boundary equilibria."""

import numpy as np
from scipy.constants import mu_0

from desc.backend import jnp
from desc.compute import (
    compute_covariant_magnetic_field,
    compute_jacobian,
    compute_magnetic_field_magnitude,
)
from desc.grid import LinearGrid
from desc.nestor import Nestor
from desc.objectives.objective_funs import _Objective
from desc.transform import Transform

from .normalization import compute_scaling_factors


def compute_I(
    R_lmn,
    Z_lmn,
    L_lmn,
    i_l,
    c_l,
    Psi,
    R_transform,
    Z_transform,
    L_transform,
    i_profile,
    c_profile,
):
    data = compute_covariant_magnetic_field(
        R_lmn,
        Z_lmn,
        L_lmn,
        i_l,
        c_l,
        Psi,
        R_transform,
        Z_transform,
        L_transform,
        i_profile,
        c_profile,
    )
    w = R_transform.grid.weights
    signgs = jnp.sign(data["sqrt(g)"].mean())
    return signgs * jnp.sum(data["B_theta"] * w) / (mu_0 * 2 * np.pi)


def bsq_vac(
    R_lmn,
    Z_lmn,
    L_lmn,
    i_l,
    c_l,
    Psi,
    R_transform,
    Z_transform,
    L_transform,
    i_profile,
    c_profile,
    nest,
):

    ctor = compute_I(
        R_lmn,
        Z_lmn,
        L_lmn,
        i_l,
        c_l,
        Psi,
        R_transform,
        Z_transform,
        L_transform,
        i_profile,
        c_profile,
    )
    out = nest.compute(R_lmn, Z_lmn, ctor)
    grid = nest._Rb_transform.grid
    bsq = out[1]["|B|^2"].reshape((grid.num_zeta, grid.num_theta)).T.flatten()
    return bsq / (2 * mu_0)


def bsq_plasma(
    R_lmn,
    Z_lmn,
    L_lmn,
    i_l,
    c_l,
    Psi,
    R_transform,
    Z_transform,
    L_transform,
    i_profile,
    c_profile,
):

    data = compute_magnetic_field_magnitude(
        R_lmn,
        Z_lmn,
        L_lmn,
        i_l,
        c_l,
        Psi,
        R_transform,
        Z_transform,
        L_transform,
        i_profile,
        c_profile,
    )
    bsq = data["|B|^2"]
    return bsq / (2 * mu_0)


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
        Bgrid = LinearGrid(rho=1, theta=self.ntheta, zeta=self.nzeta, NFP=eq.NFP)
        self.BR_transform = Transform(Bgrid, eq.R_basis, derivs=1)
        self.BZ_transform = Transform(Bgrid, eq.Z_basis, derivs=1)
        self.BL_transform = Transform(Bgrid, eq.L_basis, derivs=1)

        Igrid = LinearGrid(theta=self.ntheta, N=0, rho=1, NFP=eq.NFP)
        self.IR_transform = Transform(Igrid, eq.R_basis, derivs=1)
        self.IZ_transform = Transform(Igrid, eq.Z_basis, derivs=1)
        self.IL_transform = Transform(Igrid, eq.L_basis, derivs=1)

        self._dim_f = Bgrid.num_nodes

        self.Bp_profile = eq.pressure.copy()
        self.Ip_profile = eq.pressure.copy()
        self.Bp_profile.grid = Bgrid
        self.Ip_profile.grid = Igrid
        if eq.iota is not None:
            self.Bi_profile = eq.iota.copy()
            self.Bi_profile.grid = Bgrid
            self.Ii_profile = eq.iota.copy()
            self.Ii_profile.grid = Igrid
            self.Bc_profile = None
            self.Ic_profile = None
        else:
            self.Bi_profile = None
            self.Ii_profile = None
            self.Bc_profile = eq.current.copy()
            self.Ic_profile = eq.current.copy()
            self.Bc_profile.grid = Bgrid
            self.Ic_profile.grid = Igrid

        if self._normalize:
            scales = compute_scaling_factors(eq)
            # local quantity, want to divide by number of nodes
            self._normalization = scales["f"] / jnp.sqrt(self._dim_f)

        super().build(eq=eq, use_jit=use_jit, verbose=verbose)

    def compute(self, R_lmn, Z_lmn, L_lmn, p_l, i_l, c_l, Psi):

        bv = bsq_vac(
            R_lmn,
            Z_lmn,
            L_lmn,
            i_l,
            c_l,
            Psi,
            self.IR_transform,
            self.IZ_transform,
            self.IL_transform,
            self.Ii_profile,
            self.Ic_profile,
            self.nest,
        )
        bp = bsq_plasma(
            R_lmn,
            Z_lmn,
            L_lmn,
            i_l,
            c_l,
            Psi,
            self.BR_transform,
            self.BZ_transform,
            self.BL_transform,
            self.Bi_profile,
            self.Bc_profile,
        )
        data = compute_jacobian(R_lmn, Z_lmn, self.BR_transform, self.BZ_transform)
        w = self.BR_transform.grid.weights
        g = data["|e_theta x e_zeta|"]
        return (bv - bp - self.Bp_profile([1.0])) * w * g
