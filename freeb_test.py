import numpy as np
import os

# os.environ["JAX_LOG_COMPILES"] = "True"
import pickle
import copy
import matplotlib
import matplotlib.pyplot as plt
import scipy
from scipy.constants import mu_0
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("../"))
import functools

import jax
import jax.numpy as jnp
from jax import jit, jacfwd

from netCDF4 import Dataset
import h5py

from desc import set_device

set_device("gpu")

from desc.backend import put
from desc.basis import FourierZernikeBasis, DoubleFourierSeries, FourierSeries
from desc.equilibrium import EquilibriaFamily, Equilibrium
from desc.grid import ConcentricGrid, LinearGrid, Grid, QuadratureGrid
from desc.io import InputReader, load
from desc.plotting import (
    plot_1d,
    plot_2d,
    plot_3d,
    plot_section,
    plot_surfaces,
    plot_comparison,
)
from desc.transform import Transform
from desc.vmec import VMECIO
from desc.derivatives import Derivative
from desc.profiles import SplineProfile
from desc.magnetic_fields import SplineMagneticField

from desc.utils import flatten_list

os.getcwd()


from desc.objectives.objective_funs import _Objective
from desc.compute import (
    compute_contravariant_basis,
    compute_contravariant_magnetic_field,
    compute_pressure,
)
from desc.utils import unpack_state
from desc.geometry.utils import rpz2xyz, rpz2xyz_vec, xyz2rpz_vec, rotation_matrix


@jax.jit
def biot_loop(re, rs, J, dV):
    """

    Parameters
    ----------
    re : ndarray, shape(n_eval_pts, 3)
        evaluation points
    rs : ndarray, shape(n_src_pts, 3)
        source points
    J : ndarray, shape(n_src_pts, 3)
        current density vector at source points
    dV : ndarray, shape(n_src_pts)
        volume element at source points
    """
    re, rs, J, dV = map(jnp.asarray, (re, rs, J, dV))
    assert J.shape == rs.shape
    JdV = J * dV[:, None]
    B = jnp.zeros_like(re)

    def body(i, B):
        r = re - rs[i, :]
        num = jnp.cross(JdV[i, :], r, axis=-1)
        den = jnp.linalg.norm(r, axis=-1) ** 3
        B = B + jnp.where(den[:, None] == 0, 0, num / den[:, None])
        return B

    return 1e-7 * jax.lax.fori_loop(0, J.shape[0], body, B)


@jax.jit
def biot_loop_periods(re, rs, J, dV, NFP):

    dphi = 2 * np.pi / NFP
    R = rotation_matrix([0, 0, 1], dphi)
    B = jnp.zeros_like(re)
    Ri = jnp.eye(3)

    def body(i, B_Ri):
        B, Ri = B_Ri
        rsi = (Ri @ rs.T).T
        Ji = (Ri @ J.T).T
        B = B + biot_loop(re, rsi, Ji, dV)
        Ri = R @ Ri
        return B, Ri

    return jax.lax.fori_loop(0, NFP, body, (B, Ri))[0]


class BoundaryErrorBS(_Objective):

    _scalar = False
    _linear = False

    def __init__(
        self,
        ext_field,
        eq=None,
        target=0,
        weight=1,
        sgrid=None,
        egrid=None,
        name="B^2 Boundary",
    ):
        super().__init__(eq, target, weight, name)
        self.ext_field = ext_field
        self.sgrid = sgrid
        self.egrid = egrid
        self._callback_fmt = "Boundary Pressure Imbalance: {:10.3e} " + "Pa"

    def build(self, eq, use_jit=True, verbose=1):

        ntheta = 4 * eq.M_grid + 1
        nzeta = 4 * eq.N_grid + 1
        if self.egrid is None:
            self.egrid = LinearGrid(M=ntheta, N=nzeta, rho=1, NFP=eq.NFP)
            self.egrid.nodes += np.array([0, np.pi / ntheta, np.pi / nzeta / eq.NFP])
        if self.sgrid is None:
            self.sgrid = LinearGrid(M=ntheta, N=nzeta, rho=1, NFP=eq.NFP)

        self._dim_f = 5 * self.egrid.num_nodes
        self.NFP = eq.NFP

        self.eR_transform = Transform(self.egrid, eq.R_basis, derivs=1)
        self.eZ_transform = Transform(self.egrid, eq.Z_basis, derivs=1)
        self.eL_transform = Transform(self.egrid, eq.L_basis, derivs=1)

        self.sR_transform = Transform(self.sgrid, eq.R_basis, derivs=1)
        self.sZ_transform = Transform(self.sgrid, eq.Z_basis, derivs=1)
        self.sL_transform = Transform(self.sgrid, eq.L_basis, derivs=1)

        self.Ks_transform = Transform(self.sgrid, eq.K_basis, derivs=1)
        self.Ke_transform = Transform(self.egrid, eq.K_basis, derivs=1)

        self.ep_profile = eq.pressure.copy()
        self.ei_profile = eq.iota.copy()
        self.sp_profile = eq.pressure.copy()
        self.si_profile = eq.iota.copy()
        self.ep_profile.grid = self.egrid
        self.ei_profile.grid = self.egrid
        self.sp_profile.grid = self.sgrid
        self.si_profile.grid = self.sgrid

        self._check_dimensions()
        self._set_dimensions(eq)
        self._set_derivatives(use_jit=use_jit)
        self._built = True

    def compute(self, R_lmn, Z_lmn, L_lmn, p_l, i_l, Psi, IGphi_mn):

        Bdata_src = compute_contravariant_magnetic_field(
            R_lmn,
            Z_lmn,
            L_lmn,
            i_l,
            Psi,
            self.sR_transform,
            self.sZ_transform,
            self.sL_transform,
            self.si_profile,
        )
        ndata_src = compute_contravariant_basis(
            R_lmn, Z_lmn, self.sR_transform, self.sZ_transform
        )
        Bdata_eval = compute_contravariant_magnetic_field(
            R_lmn,
            Z_lmn,
            L_lmn,
            i_l,
            Psi,
            self.eR_transform,
            self.eZ_transform,
            self.eL_transform,
            self.ei_profile,
        )
        ndata_eval = compute_contravariant_basis(
            R_lmn, Z_lmn, self.eR_transform, self.eZ_transform
        )
        pdata_eval = compute_pressure(p_l, self.ep_profile)

        I = IGphi_mn[0] / mu_0
        G = IGphi_mn[1] / mu_0
        phi_mn = IGphi_mn[2:] / mu_0

        K_t = self.Ks_transform.transform(phi_mn, dt=1)[:, np.newaxis] + I / (2 * np.pi)
        K_z = self.Ks_transform.transform(phi_mn, dz=1)[:, np.newaxis] + G / (2 * np.pi)
        Ks = K_t * ndata_src["e_theta"] + K_z * ndata_src["e_zeta"]
        Ks = rpz2xyz_vec(Ks, phi=self.Ks_transform.grid.nodes[:, 2])
        K_t = self.Ke_transform.transform(phi_mn, dt=1)[:, np.newaxis] + I / (2 * np.pi)
        K_z = self.Ke_transform.transform(phi_mn, dz=1)[:, np.newaxis] + G / (2 * np.pi)
        Ke = K_t * ndata_eval["e_theta"] + K_z * ndata_eval["e_zeta"]
        Ke = rpz2xyz_vec(Ke, phi=self.Ke_transform.grid.nodes[:, 2])

        Bsrc = Bdata_src["B"]
        Bsrc = rpz2xyz_vec(Bsrc, phi=self.sR_transform.grid.nodes[:, 2])
        nsrc = (
            ndata_src["e^rho"]
            / jnp.linalg.norm(ndata_src["e^rho"], axis=1)[:, np.newaxis]
        )
        nsrc = rpz2xyz_vec(nsrc, phi=self.sR_transform.grid.nodes[:, 2])
        J = jnp.cross(nsrc, Bsrc, axis=1) / mu_0

        neval = (
            ndata_eval["e^rho"]
            / jnp.linalg.norm(ndata_eval["e^rho"], axis=1)[:, np.newaxis]
        )
        neval = rpz2xyz_vec(neval, phi=self.eR_transform.grid.nodes[:, 2])

        rsrc = jnp.array(
            [Bdata_src["R"], self.sR_transform.grid.nodes[:, 2], Bdata_src["Z"]]
        ).T
        rsrc = rpz2xyz(rsrc)
        reval = jnp.array(
            [Bdata_eval["R"], self.eR_transform.grid.nodes[:, 2], Bdata_eval["Z"]]
        ).T
        reval = rpz2xyz(reval)

        dVeval = abs(
            Bdata_eval["|e_theta x e_zeta|"] * self.eR_transform.grid.weights
        )  # TODO: change to spacing
        dVsrc = abs(
            Bdata_src["|e_theta x e_zeta|"] * self.sR_transform.grid.weights
        )  # TODO: change to spacing

        B_tot = rpz2xyz_vec(Bdata_eval["B"], phi=self.eR_transform.grid.nodes[:, 2])
        B_plasma = (
            biot_loop_periods(reval, rsrc, J, dVsrc / self.NFP, self.NFP) + B_tot / 2
        )
        B_sheet = biot_loop_periods(reval, rsrc, Ks, dVsrc / self.NFP, self.NFP)
        B_ex = self.ext_field.compute_magnetic_field(reval, basis="xyz")

        B_in = B_tot
        B_out = B_ex + B_plasma + B_sheet
        Bsq_in = jnp.sum(B_in ** 2, axis=-1) + 2 * mu_0 * pdata_eval["p"]
        Bsq_out = jnp.sum(B_out ** 2, axis=-1)
        Bsq_diff = (Bsq_in - Bsq_out) * dVeval
        Bn = jnp.sum(B_out * neval, axis=-1) * dVeval
        Kerr = B_out - (B_in + mu_0 * jnp.cross(neval, Ke, axis=1))
        return jnp.concatenate([Bsq_diff, Bn, Kerr.flatten()])


f = Dataset("wout_test_beta.vmec.nc")
print(f["mpol"][:])
print(f["ntor"][:])


veq = VMECIO.load("wout_test_beta.vmec.nc", spectral_indexing="fringe")
veq.change_resolution(L=16, M=8, N=8, L_grid=20, M_grid=12, N_grid=12)

pres = np.asarray(f.variables["presf"])
sp = np.linspace(0, 1, pres.size)
rp = np.sqrt(sp)
pressure = SplineProfile(pres, rp)

iot = np.asarray(f.variables["iotaf"])
si = np.linspace(0, 1, iot.size)
ri = np.sqrt(si)
iota = SplineProfile(iot, ri)


veq.pressure = pressure
veq.iota = iota

NFP = veq.NFP
mgrid = "tests/inputs/nestor/mgrid_test.nc"
extcur = f["extcur"][:]
folder = os.getcwd()
mgridFilename = os.path.join(folder, mgrid)
ext_field = SplineMagneticField.from_mgrid(
    mgridFilename, extcur, extrap=True, period=(2 * np.pi / NFP)
)


veq.resolution_summary()
veq.solve(ftol=1e-2, xtol=1e-6, gtol=1e-6, maxiter=1000, verbose=3)


surf = veq.get_surface_at(1)
surf.change_resolution(M=1, N=0)
eq = Equilibrium(
    Psi=veq.Psi,
    pressure=veq.pressure,
    iota=veq.iota,
    spectral_indexing=veq.spectral_indexing,
    sym=veq.sym,
    NFP=veq.NFP,
)

surf.Z_lmn = surf.R_lmn[-1:]

eq.set_initial_guess(surf)
eq.surface = surf

eq.change_resolution(veq.L, veq.M, veq.N, veq.L_grid, veq.M_grid, veq.N_grid)
eq.solve(ftol=1e-2, verbose=3)

from desc.objectives import (
    ObjectiveFunction,
    ForceBalance,
    LambdaGauge,
    FixPressure,
    FixIota,
    FixPsi,
)

bc_objective = BoundaryErrorBS(ext_field)
fb_objective = ForceBalance()

objective = ObjectiveFunction(bc_objective)
constraints = (
    fb_objective,
    FixPressure(),
    FixIota(),
    FixPsi(),
)

fb_objective.build(eq)
bc_objective.build(eq)


def print_error_summary(eqis):
    for eqi in eqis:
        f = fb_objective.callback(
            eqi.R_lmn, eqi.Z_lmn, eqi.L_lmn, eqi.p_l, eqi.i_l, eqi.Psi, eq.IGphi_mn
        )
        b = bc_objective.callback(
            eqi.R_lmn, eqi.Z_lmn, eqi.L_lmn, eqi.p_l, eqi.i_l, eqi.Psi, eq.IGphi_mn
        )


bc_objective.callback(eq.R_lmn, eq.Z_lmn, eq.L_lmn, eq.p_l, eq.i_l, eq.Psi, eq.IGphi_mn)
bc_objective.callback(
    veq.R_lmn, veq.Z_lmn, veq.L_lmn, veq.p_l, veq.i_l, veq.Psi, eq.IGphi_mn
)

eq1 = eq.copy()
out = eq1.optimize(
    objective,
    constraints,
    maxiter=60,
    verbose=3,
    options={
        "perturb_options": {"order": 2},
        "initial_trust_radius": 1e-3,
        "ga_tr_ratio": 0,
    },
)

eq1.save("freeb_test_out.h5")
with open("freeb_beta_out.pkl", "wb+") as f:
    pickle.dump(out, f)

out = veq.optimize(
    objective,
    constraints,
    maxiter=60,
    verbose=3,
    options={
        "perturb_options": {"order": 2},
        "initial_trust_radius": 1e-3,
        "ga_tr_ratio": 0,
    },
)

veq.save("freeb_test_outv.h5")
with open("freeb_beta_outv.pkl", "wb+") as f:
    pickle.dump(out, f)
