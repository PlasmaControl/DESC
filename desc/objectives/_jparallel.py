"""Objectives on the parallel adiabatic invariant J|| at fixed bounce fields.

J||(ρ, α; B*) = ∫ √(softplus(1 − B/B*)) / B dζ_B over one field period on the Boozer
field line ϑ_B = α + ι ζ_B. The bounce fields B*(t) = B_min + t (B_max − B_min) are
taken on one reference surface ρ_ref and held fixed over the surfaces, so an orbit
that drifts across surfaces keeps its absolute bounce field and the J|| contours are
its orbit-averaged paths. ∂J||/∂α is computed analytically through B.

|B| on the field lines is summed per poloidal mode,
B = Σ_k cos(kα) P_k(ζ) + sin(kα) Q_k(ζ), never as a (lines × points) × modes basis
matrix, which needs ~100 GB at the resolutions used for stellarator-mirror hybrids.
"""

import numpy as np

from desc.backend import jax, jnp
from desc.compute import get_profiles, get_transforms
from desc.compute._omnigenity import (
    _reshape_surface_coefficients,
    _smoothmax_logsumexp,
    _softplus_relu,
    _softplus_relu_sigmoid,
    _trapz,
)
from desc.compute.utils import _compute as compute_fun
from desc.grid import LinearGrid

from .objective_funs import _Objective


class _JparFixedB(_Objective):
    """Common part: J|| and ∂J||/∂α on (t, ρ, α) at the fixed bounce fields."""

    _coordinates = ""
    _units = "(dimensionless)"
    _static_attrs = _Objective._static_attrs + [
        "_num_alpha",
        "_nzeta",
        "_sharp",
        "_ext_tau",
        "M_booz",
        "N_booz",
        "_iref",
        "_K",
        "_nmn",
    ]

    def __init__(
        self,
        eq,
        rho,
        t,
        rho_ref=0.5,
        num_alpha=32,
        nzeta=1000,
        softplus_sharpness=2000.0,
        soft_extrema_tau=1e-3,
        M_booz=None,
        N_booz=None,
        weight=1,
        deriv_mode="auto",
        jac_chunk_size=None,
        name="J||",
    ):
        self._rho, self._t = np.asarray(rho, float), np.asarray(t, float)
        assert np.min(np.abs(self._rho - rho_ref)) < 1e-12, "rho must contain rho_ref"
        self._iref = int(np.argmin(np.abs(self._rho - rho_ref)))
        self._num_alpha, self._nzeta = num_alpha, nzeta
        self._sharp, self._ext_tau = softplus_sharpness, soft_extrema_tau
        self.M_booz = 2 * eq.M if M_booz is None else M_booz
        self.N_booz = 2 * eq.N if N_booz is None else N_booz
        super().__init__(
            things=eq,
            target=0,
            bounds=None,
            weight=weight,
            normalize=False,
            normalize_target=False,
            deriv_mode=deriv_mode,
            jac_chunk_size=jac_chunk_size,
            name=name,
        )

    def _build_common(self):
        eq = self.things[0]
        grid = LinearGrid(
            rho=self._rho, M=2 * self.M_booz, N=2 * self.N_booz, NFP=eq.NFP, sym=False
        )
        keys = ["|B|_mn_B", "iota"]
        tr = get_transforms(
            keys, obj=eq, grid=grid, M_booz=self.M_booz, N_booz=self.N_booz
        )
        # (l, m, n) rows: m >= 0 cos(m theta), m < 0 sin(|m| theta); n likewise
        modes = tr["B"].basis.modes
        self._K = int(np.max(np.abs(modes[:, 1])))
        Nn = int(np.max(np.abs(modes[:, 2])))
        zeta = np.linspace(0, 2 * np.pi / eq.NFP, self._nzeta)
        nn = np.arange(-Nn, Nn + 1)
        Fn = np.where(
            nn[:, None] >= 0,
            np.cos(np.abs(nn)[:, None] * eq.NFP * zeta),
            np.sin(np.abs(nn)[:, None] * eq.NFP * zeta),
        )
        alpha = np.linspace(0, 2 * np.pi, self._num_alpha, endpoint=False)
        k = np.arange(self._K + 1)
        self._constants = {
            "transforms": tr,
            "profiles": get_profiles(keys, obj=eq, grid=grid),
            "t": jnp.asarray(self._t),
            # soft extrema of the reference surface
            "Agrid": jnp.asarray(tr["B"].basis.evaluate(grid.nodes[:: grid.num_rho])),
            "mi": jnp.asarray(modes[:, 1] + self._K),
            "ni": jnp.asarray(modes[:, 2] + Nn),
            "Fn": jnp.asarray(Fn),
            "zeta": jnp.asarray(zeta),
            "ca": jnp.asarray(np.cos(np.outer(alpha, k))),
            "sa": jnp.asarray(np.sin(np.outer(alpha, k))),
            "k": jnp.asarray(k, dtype=float),
        }
        self._nmn = (2 * self._K + 1, 2 * Nn + 1)

    def _surface_data(self, params, c):
        grid = c["transforms"]["grid"]
        data = compute_fun(
            "desc.equilibrium.equilibrium.Equilibrium",
            ["|B|_mn_B", "iota"],
            params=params,
            transforms=c["transforms"],
            profiles=c["profiles"],
        )
        coeff = _reshape_surface_coefficients(grid, data["|B|_mn_B"])  # (rho, modes)
        iota = grid.compress(data["iota"])
        Bref = coeff[self._iref] @ c["Agrid"].T
        tau = self._ext_tau * (jnp.max(Bref) - jnp.min(Bref)) / jnp.log(Bref.size)
        bmax = _smoothmax_logsumexp(Bref, axis=0, tau=tau)
        bmin = -_smoothmax_logsumexp(-Bref, axis=0, tau=tau)
        return coeff, iota, bmin, bmax

    def _field_lines(self, cf, io, c):
        """|B| and ∂|B|/∂α on (α, ζ) of one surface."""
        K, zeta, k = self._K, c["zeta"], c["k"]
        # G_m(zeta) = sum_n c_mn F_n(zeta); m = 0..K and their sin partners -m
        G = jnp.zeros(self._nmn).at[c["mi"], c["ni"]].add(cf) @ c["Fn"]
        Gp, Gm = G[K:], jnp.concatenate([jnp.zeros_like(G[:1]), G[:K][::-1]])
        ck, sk = jnp.cos(k[:, None] * io * zeta), jnp.sin(k[:, None] * io * zeta)
        P, Q = ck * Gp + sk * Gm, ck * Gm - sk * Gp
        B = c["ca"] @ P + c["sa"] @ Q
        dB = c["ca"] @ (k[:, None] * Q) - c["sa"] @ (k[:, None] * P)
        return B, dB

    def jmaps(self, params, constants):
        """J|| and ∂J||/∂α, each of shape (t, rho, alpha)."""
        c = constants
        coeff, iota, bmin, bmax = self._surface_data(params, c)
        Bstar = bmin + c["t"] * (bmax - bmin)
        zeta = c["zeta"]

        def surface(x):
            B, dB = self._field_lines(x[0], x[1], c)
            arg = 1.0 - B[None] / Bstar[:, None, None]
            cut = _softplus_relu(arg, beta=self._sharp)
            cut = jnp.where(cut > 1e-200, cut, 0.0)
            sq = jnp.sqrt(jnp.maximum(cut, 1e-200))
            safe = jnp.where(cut > 0, sq, 1e-30)
            dfdB = jnp.where(
                cut > 0,
                -_softplus_relu_sigmoid(arg, beta=self._sharp)
                / (2.0 * Bstar[:, None, None] * safe * B[None])
                - safe / B[None] ** 2,
                0.0,
            )
            J = _trapz(jnp.where(cut > 0, sq, 0.0) / B[None], zeta, axis=-1)
            return J, _trapz(dfdB * dB[None], zeta, axis=-1)

        J, dJ = jax.lax.map(surface, (coeff, iota))  # (rho, t, alpha) each
        return jnp.transpose(J, (1, 0, 2)), jnp.transpose(dJ, (1, 0, 2))


class JparDjdaNorm(_JparFixedB):
    """Normalized ∂J||/∂α at fixed bounce fields.

    Residual per (t, ρ in [rho_lo, rho_hi], α):
    (∂J||/∂α) / (|⟨J||⟩_α(ρ)| + floor |⟨J||⟩_α(ρ_ref)|) / √(n_ρ n_α).

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium to be optimized.
    rho : array-like
        Surfaces; must contain ``rho_ref``.
    t : array-like
        Bounce-field levels in [0, 1] between B_min and B_max of ``rho_ref``.
    rho_lo, rho_hi : float
        Range of surfaces whose residuals are returned.
    floor : float
        Fraction of the reference surface's ⟨J||⟩ added to the normalization.

    """

    _print_value_fmt = "normalized dJ||/dalpha (fixed B*): "

    def __init__(
        self,
        eq,
        rho,
        t,
        rho_lo=0.35,
        rho_hi=0.98,
        floor=0.1,
        name="J|| dJ/dalpha normalized",
        **kw,
    ):
        r = np.asarray(rho)
        self._sel = np.where((r >= rho_lo - 1e-9) & (r <= rho_hi + 1e-9))[0]
        self._floor = floor
        super().__init__(eq, rho, t, name=name, **kw)

    def build(self, use_jit=True, verbose=1):
        """Build constant arrays."""
        self._build_common()
        self._dim_f = self._t.size * self._sel.size * self._num_alpha
        super().build(use_jit=use_jit, verbose=verbose)

    def _normalized(self, params, constants):
        c = self.constants if constants is None else constants
        J, dJ = self.jmaps(params, c)
        m = jnp.sqrt(J.mean(-1) ** 2 + 1e-60)  # (t, rho)
        den = m[:, self._sel] + self._floor * m[:, self._iref][:, None]
        return dJ[:, self._sel] / den[..., None]  # (t, rho, alpha)

    def compute(self, params, constants=None):
        """Compute the normalized ∂J||/∂α residuals."""
        r = self._normalized(params, constants)
        return jnp.ravel(r) / np.sqrt(self._sel.size * self._num_alpha)


class JparDjdaTopK(JparDjdaNorm):
    """``JparDjdaNorm`` keeping only the ``top_k`` worst field lines per (t, ρ).

    J|| is still computed on all ``num_alpha`` lines; the residuals (signed) are the
    ``top_k`` of largest magnitude, so the Jacobian has t × ρ × top_k rows.
    """

    _static_attrs = JparDjdaNorm._static_attrs + ["_top_k"]

    def __init__(self, eq, rho, t, top_k=10, name="J|| dJ/dalpha normalized", **kw):
        self._top_k = int(top_k)
        super().__init__(eq, rho, t, name=name, **kw)

    def build(self, use_jit=True, verbose=1):
        """Build constant arrays."""
        self._build_common()
        self._dim_f = self._t.size * self._sel.size * self._top_k
        _Objective.build(self, use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute the top-k normalized ∂J||/∂α residuals."""
        r = self._normalized(params, constants)
        _, idx = jax.lax.top_k(jnp.abs(r), self._top_k)
        out = jnp.take_along_axis(r, idx, axis=-1)
        return jnp.ravel(out) / np.sqrt(self._sel.size * self._top_k)


class JparOrder(_JparFixedB):
    """Closure of the fixed-B* J|| contours about the axis out to the rim ``rho[-1]``.

    Residual per (t, inner surface ρ_i):
    softplus(k (softmax_α d J||(ρ_i) − softmin_α d J||(rim)) / span + margin) / k,
    zero once every line of ρ_i lies inside the rim level, so the contours through
    it close about the axis. d is the sign of the J|| trend from the axis to the
    rim, fixed at build, as are the soft-extremum temperatures.
    """

    _print_value_fmt = "J|| closure order (fixed B*): "

    def __init__(
        self,
        eq,
        rho,
        t,
        sharpness=20.0,
        tau=0.02,
        margin=0.0,
        name="J|| closure order",
        **kw,
    ):
        self._k_or, self._tau_or, self._margin = sharpness, tau, margin
        super().__init__(eq, rho, t, name=name, **kw)

    def build(self, use_jit=True, verbose=1):
        """Build constant arrays."""
        self._build_common()
        J = np.asarray(self.jmaps(self.things[0].params_dict, self._constants)[0])
        d = np.where(J[:, -1].mean(-1) >= J[:, 0].mean(-1), 1.0, -1.0)
        dJ = d[:, None, None] * J
        T = self._tau_or * (dJ.max(-1) - dJ.min(-1)) + 1e-12 * np.abs(dJ).max()
        self._constants.update(d=jnp.asarray(d), T=jnp.asarray(T))
        self._dim_f = self._t.size * (self._rho.size - 1)
        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute the closure residuals."""
        c = self.constants if constants is None else constants
        dJ = c["d"][:, None, None] * self.jmaps(params, c)[0]
        T = c["T"]
        top = _smoothmax_logsumexp(dJ[:, :-1], axis=-1, tau=T[:, :-1, None]).squeeze(-1)
        low = -_smoothmax_logsumexp(-dJ[:, -1], axis=-1, tau=T[:, -1:]).squeeze(-1)
        span = jnp.abs(dJ[:, -1].mean(-1) - dJ[:, 0].mean(-1)) + 1e-30
        gap = (top - low[:, None]) / span[:, None] + self._margin
        out = jnp.logaddexp(0.0, self._k_or * gap) / self._k_or
        return jnp.ravel(out) / np.sqrt(self._rho.size - 1)


class JparSingleWell(_JparFixedB):
    """One |B| well at the bottom of every field line.

    Per (ρ, α): the |B| rise on the wrong side of the line's own minimum, weighted to
    B < B_cut = B_min + t_cut (B_max − B_min) of ρ_ref, summed over the period and
    divided by B_max − B_min of ρ_ref (the height of a second-well hump in units of
    the well depth; zero for a single well). The minimum's position is held fixed
    (stop_gradient) within each evaluation.
    """

    _print_value_fmt = "second |B| well at the bottom (hump / well depth): "

    def __init__(self, eq, rho, t_cut=0.3, beta=200.0, name="single |B| well", **kw):
        self._t_cut, self._beta = t_cut, beta
        super().__init__(eq, rho, np.array([t_cut]), name=name, **kw)

    def build(self, use_jit=True, verbose=1):
        """Build constant arrays."""
        self._build_common()
        self._dim_f = self._rho.size * self._num_alpha
        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute the second-well residuals."""
        c = self.constants if constants is None else constants
        coeff, iota, bmin, bmax = self._surface_data(params, c)
        Bcut = bmin + self._t_cut * (bmax - bmin)
        zeta = c["zeta"]
        zmid = 0.5 * (zeta[1:] + zeta[:-1])

        def surface(x):
            B = self._field_lines(x[0], x[1], c)[0]  # (alpha, zeta)
            z0 = zeta[jnp.argmin(jax.lax.stop_gradient(B), axis=1)]
            # -1 left of the minimum (B must fall), +1 right of it (B must rise)
            s = jnp.sign(zmid[None, :] - z0[:, None])
            dB = jnp.diff(B, axis=1)
            w = jax.nn.sigmoid(self._beta * (1.0 - 0.5 * (B[:, 1:] + B[:, :-1]) / Bcut))
            return jnp.sum(w * jax.nn.relu(-s * dB), axis=1) / (bmax - bmin)

        out = jax.lax.map(surface, (coeff, iota))
        return jnp.ravel(out) / np.sqrt(self._rho.size * self._num_alpha)
