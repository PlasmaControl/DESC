"""The tied λ of a RidgeEquilibrium: its ring quadrature and its compute functions.

A ``RidgeEquilibrium`` has R, Z on a ``FourierZernikeRidgeBasis``, whose boundary has
m_b corners on the X-lines α_b = θ - ι_b ζ = 2πk/m_b. Its stream function is tied to the
Jacobian,

    λ = λ_0 + λ_1 - E,   λ_1 = S / G,   S = A_θ[√g - G],   G = <√g>_θ,

with <.>_θ the average over the ring of fixed (ρ, ζ), A_θ the zero-mean antiderivative
in θ along it, λ_0 the ordinary Fourier-Zernike series (``L_lmn``) and E the ridge
subtraction (``RidgeSubtraction``). Then 1 + ∂_θ λ_1 = √g / G, so λ follows the corner
structure of √g, and E makes λ constant along the X-lines. A_θ commutes with ∂_ρ and
∂_ζ, so S_x = A_θ[√g_x - G_x].

The rings are evaluated by ``TiedLambdaRings``, the ``"ring"`` transform that
``get_transforms`` builds for a grid. The λ keys registered here for the
``RidgeEquilibrium`` replace those of the ``Equilibrium``, so everything computed for a
RidgeEquilibrium (``ForceBalance``, ``eq.compute``, the plotting functions) uses the
tied λ. Only the λ keys that ``|B|``, ``|J|`` and the force need are provided; the
others raise.
"""

import functools
from math import comb, gcd

import numpy as np

from desc.backend import jnp
from desc.basis import disc_ridge_function, fourier, zernike_radial
from desc.grid import Grid
from desc.io import IOAble
from desc.utils import errorif

from .data_index import data_index, register_compute_fun

_EQ = "desc.equilibrium.equilibrium.Equilibrium"
_RING_KEYS = ["sqrt(g)", "sqrt(g)_r", "sqrt(g)_z", "sqrt(g)_rz", "sqrt(g)_zz"]


# ------------------------------------------------------------------------------------
# the rings: a calibrated rule, clustered at the corners
# ------------------------------------------------------------------------------------
RING_B = 0.5  # the fraction of the ring nodes in the clustered part of the map s(α)
_RING_SIZES = (48, 64, 80, 96, 128, 160, 192)  # the calibrated sizes of clustered rings


def default_ring_rule(rho, M):
    """The ring of the tied λ at the radius ``rho`` for the resolution ``M``: (n, eps).

    * ρ > 0.7: clustered, eps = 3 (1 - ρ) (clipped to [5e-3, 0.5]), and n the smallest
      of 48, 64, 80, 96, 128, 160, 192 (then steps of 32) that is at least
      16 + 28 ln(1/(1 - ρ)) and, for ρ > 0.99, at least 20/sqrt(1 - ρ).
    * ρ ≤ 0.7: uniform (eps = None), n = max(24, 8 ceil(2.2 M ρ²)).

    Calibrated on the W7-X tied solves at M = 7, 9 (and ring-only checks at M = 11, 13)
    as the smallest rule with a ring error ≤ 1e-2 (max) and ≤ 1e-3 (median) of the
    local force at every collocation level, against a converged clustered reference.

    Parameters
    ----------
    rho : float
        Radius of the ring.
    M : int
        Poloidal resolution of the equilibrium.

    Returns
    -------
    n : int
        Number of ring nodes.
    eps : float or None
        Clustering width of ``s_of_a``, None for a uniform ring.

    """
    rho = float(rho)
    if rho > 0.7:
        raw = 16.0 + 28.0 * np.log(1.0 / max(1.0 - rho, 1e-300))
        if rho > 0.99:
            raw = max(raw, 20.0 / np.sqrt(1.0 - rho))
        n = next((k for k in _RING_SIZES if k >= raw), None)
        if n is None:
            n = int(_RING_SIZES[-1] + 32 * np.ceil((raw - _RING_SIZES[-1]) / 32))
        return n, float(np.clip(3.0 * (1 - rho), 5e-3, 0.5))
    return int(max(24, 8 * np.ceil(2.2 * M * rho**2))), None


def s_of_a(a, eps, m_b, b=RING_B):
    """The clustering map s(α) = (1 - b) α + b (2/m_b) arctan(tan(m_b α/2) / eps).

    The branch of arctan is chosen so that s is continuous and increasing, with
    s(α + 2π/m_b) = s(α) + 2π/m_b and s(-α) = -s(α). Nodes uniform in s cluster at the
    corners α = 2πk/m_b over a width ~eps (in m_b α/2).
    """
    u = m_b * np.asarray(a, dtype=float) / 2
    base = np.round(u / np.pi) * np.pi
    return (1 - b) * np.asarray(a) + b * (2 / m_b) * (
        base + np.arctan(np.tan(u - base) / eps)
    )


def ds_of_a(a, eps, m_b, b=RING_B):
    """ds/dα of ``s_of_a``."""
    u = m_b * np.asarray(a, dtype=float) / 2
    return (1 - b) + b * eps / (eps**2 * np.cos(u) ** 2 + np.sin(u) ** 2)


def a_of_s(s, eps, m_b, b=RING_B):
    """The inverse of ``s_of_a`` on [0, 2π): a dense table, then Newton's method."""
    A = np.linspace(0, 2 * np.pi, 2**21)
    Sa = s_of_a(A, eps, m_b, b)
    Sa[0], Sa[-1] = 0.0, 2 * np.pi
    s = np.asarray(s, dtype=float)
    a = np.interp(s, Sa, A)
    for _ in range(4):
        a = a - (s_of_a(a, eps, m_b, b) - s) / ds_of_a(a, eps, m_b, b)
    return a


def _ring_theta(n):
    """Uniform ring of n nodes offset by half a spacing, so no node sits on a corner."""
    return 2 * np.pi * (np.arange(n) + 0.5) / n


def _dft_antiderivative_kernel(theta_c, n):
    """Kernel K of the zero-mean antiderivative on a uniform ring.

    S(θ_c) = K @ f is the zero-mean antiderivative of the trigonometric interpolant of
    f at the nodes θ_j = ``_ring_theta(n)`` (the Nyquist mode dropped):

        K[c, j] = (2/n) Σ_{m=1}^{n/2-1} sin(m (θ_c - θ_j)) / m,

    a discrete Fourier transform in m for every target, done by the FFT.
    """
    m = np.arange(n)
    b = np.where((m > 0) & (m < n // 2), 1.0 / np.maximum(m, 1), 0.0)
    C = b[None, :] * np.exp(
        1j * m[None, :] * (np.asarray(theta_c)[:, None] - np.pi / n)
    )
    return 2 / n * np.imag(np.fft.fft(C, axis=1))


def clustered_ring_matrices(a_c, n, eps, m_b, b=RING_B):
    """The quadrature of one ring, for the targets at α = ``a_c``.

    With the nodes α_j = s^-1(2π (j + 1/2)/n) and the values f_j = f(α_j),

        G = wG @ f,    S(a_c) = A @ f - G rowA,

    are the ring mean of f and the zero-mean antiderivative in α of f - G at the
    targets: wG_j = φ'_j / n and A = [K_n(s_c - s_j) - κ_j] φ'_j, with φ' = 1/s'(α),
    K_n the kernel of ``_dft_antiderivative_kernel``, s_c = s(a_c) and
    κ_j = (1/n) Σ_i φ'_i K_n(s_i - s_j), so that S has zero mean in α. ``eps=None``
    gives a uniform ring.

    Returns
    -------
    alpha, wG, A, rowA : ndarray
        The ring nodes α_j, the weights of the mean, the antiderivative matrix and
        its row sums.

    """
    a, dphi, kappa = _ring_nodes(int(n), eps, int(m_b), b)
    a_c = np.mod(np.asarray(a_c, dtype=float), 2 * np.pi)
    s_c = a_c if eps is None else s_of_a(a_c, eps, m_b, b)
    A = (_dft_antiderivative_kernel(s_c, n) - kappa[None, :]) * dphi[None, :]
    return a, dphi / n, A, A.sum(axis=1)


@functools.lru_cache(maxsize=512)
def _ring_nodes(n, eps, m_b, b):
    """Ring nodes α_j, φ'_j = 1/s'(α_j) and κ_j of ``clustered_ring_matrices``.

    They only depend on (n, eps), so they are cached.
    """
    s = _ring_theta(n)
    if eps is None:
        a, dphi = s, np.ones(n)
    else:
        a = a_of_s(s, eps, m_b, b)
        dphi = 1 / ds_of_a(a, eps, m_b, b)
    kappa = (dphi @ _dft_antiderivative_kernel(s, n)) / n
    for x in (a, dphi, kappa):
        x.setflags(write=False)
    return a, dphi, kappa


# ------------------------------------------------------------------------------------
# the separable ring evaluator
# ------------------------------------------------------------------------------------
def _deriv_key(X, d):
    """The DESC data key of the derivative d = (a, b, c) of X ("R" or "Z")."""
    s = "r" * d[0] + "t" * d[1] + "z" * d[2]
    return X + ("_" + s if s else "")


class SeparableRingEvaluator(IOAble):
    """R, Z and their derivatives on the sheared rings of one radius.

    The ring nodes are (ρ, α_j + δ_k, ζ_k), in the order k n + j (k over ``zetas``,
    j over ``alpha``), with δ_k = ``shifts`` (default ι_b ζ_k). As δ_k - ι_b ζ_k is a
    multiple of 2π/m_b, the period of the ridge factor, the helical angle of every node
    is α_j modulo that period, and the ridge factor q_s(ρ, α_j) (q_0 = 1,
    q_1 = Re Q_γ, q_2 = Im Q_γ) is computed once per radius. A mode
    q_s R_l^|m|(ρ) f_m(θ) g_n(ζ) of the ``FourierZernikeRidgeBasis`` then splits with
    f_m(α + δ) = f_m(α) cos(|m| δ) + f^_m(α) sin(|m| δ), f^_m = f_m'/|m|, and each field
    is three small contractions,

        radial:    r_mns    = Σ_l c_lmns ∂_ρ^a2 R_l^|m|(ρ),
        toroidal:  T_msu(k) = C^u_m(δ_k) Σ_n ∂_ζ^c2 g_n(ζ_k) r_mns,
        poloidal:  Y_s(k,j) = Σ_mu ∂_α^b2 F^u_m(α_j) T_msu(k),

    with (C^0, C^1) = (cos, sin)(|m| δ) and (F^0, F^1) = (f_m, f^_m), followed by the
    Leibniz sum over the derivatives of q_s, where ∂_θ q = ∂_α q and
    ∂_ζ q = -ι_b ∂_α q. The constants are O(n M + n_ζ N) per radius, and ``compute`` is
    linear in R_lmn, Z_lmn.

    Parameters
    ----------
    eq : RidgeEquilibrium
        The equilibrium, whose R and Z bases are ``FourierZernikeRidgeBasis``.
    rho : float
        Radius of the rings.
    alpha : ndarray, shape(n,)
        Helical angles α_j of the ring nodes.
    zetas : ndarray, shape(n_ζ,)
        Toroidal angles ζ_k of the rings.
    shifts : ndarray, shape(n_ζ,), optional
        Poloidal shift δ_k of each ring, equal to ι_b ζ_k modulo 2π/m_b. Default
        ι_b ζ_k.
    derivs : dict, optional
        Derivative orders {"R": [(a, b, c), ...], "Z": [...]} to return. Default the
        orders of the ring keys √g, √g_r, √g_z, √g_rz, √g_zz.

    """

    _static_attrs = ["_meta", "_derivs"]

    def __init__(self, eq, rho, alpha, zetas, shifts=None, derivs=None):
        from .utils import get_derivs

        rho = float(rho)
        alpha = np.asarray(alpha, dtype=float)
        zeta = np.asarray(zetas, dtype=float)
        iota_b = eq.iota_b
        delta = iota_b * zeta if shifts is None else np.asarray(shifts, dtype=float)
        k = (delta - iota_b * zeta) / (2 * np.pi / eq.m_b)
        errorif(
            not np.allclose(k, np.round(k), rtol=0, atol=1e-9),
            ValueError,
            "ring shifts must equal iota_b zeta modulo 2 pi/m_b",
        )
        if derivs is None:
            derivs = get_derivs(_RING_KEYS, _EQ)
        self._derivs = {
            X: tuple(sorted({tuple(int(v) for v in d) for d in derivs[X]}))
            for X in ("R", "Z")
            if X in derivs
        }
        self._meta, self._constants = {}, {}
        for X, ders in self._derivs.items():
            basis = getattr(eq, X + "_basis")
            l, m, n = basis.modes.T
            s = basis.blocks
            mv, nv = np.unique(m), np.unique(n)
            im, jn = np.searchsorted(mv, m), np.searchsorted(nv, n)
            ## the gather map (m, n, s, slot) -> mode; the empty slots point to an extra
            ## zero coefficient at the end
            key = (im * nv.size + jn) * 3 + s
            counts = np.bincount(key, minlength=mv.size * nv.size * 3)
            order = np.argsort(key, kind="stable")
            start = np.concatenate([[0], np.cumsum(counts)[:-1]])
            slot = np.empty(len(key), dtype=int)
            slot[order] = np.arange(len(key)) - start[key[order]]
            idx = np.full((mv.size, nv.size, 3, max(1, counts.max())), len(key))
            idx[im, jn, s, slot] = np.arange(len(key))

            ## every lower derivative that the Leibniz sum needs
            D = sorted(
                {
                    (i, j, k)
                    for a, b, c in ders
                    for i in range(a + 1)
                    for j in range(b + 1)
                    for k in range(c + 1)
                }
            )
            a2s = sorted({d[0] for d in D})
            b2s = sorted({d[1] for d in D})
            c2s = sorted({d[2] for d in D})
            rad = np.stack(
                [
                    np.append(np.asarray(zernike_radial(rho, l, m, dr=a2)), 0.0)[idx]
                    for a2 in a2s
                ]
            )
            g = np.stack(
                [
                    np.asarray(fourier(zeta[:, None], nv, NFP=eq.NFP, dt=c2))
                    for c2 in c2s
                ]
            )
            ma = np.abs(mv)
            C = np.stack(
                [
                    np.cos(ma[None, :] * delta[:, None]),
                    np.sin(ma[None, :] * delta[:, None]),
                ],
                axis=-1,
            )
            safe = np.where(ma == 0, 1, ma)
            F = np.stack(
                [
                    np.stack(
                        [
                            np.asarray(fourier(alpha[:, None], mv, dt=b2)),
                            np.where(
                                ma == 0,
                                0.0,
                                np.asarray(fourier(alpha[:, None], mv, dt=b2 + 1))
                                / safe,
                            ),
                        ],
                        axis=-1,
                    )
                    for b2 in b2s
                ]
            )
            ## the Leibniz terms of every derivative: (coefficient, derivative of q,
            ## derivative of the Fourier-Zernike factor)
            terms, qpairs = {}, set()
            for a, b, c in ders:
                terms[(a, b, c)] = []
                for a1 in range(a + 1):
                    for b1 in range(b + 1):
                        for c1 in range(c + 1):
                            coef = comb(a, a1) * comb(b, b1) * comb(c, c1)
                            coef *= (-iota_b) ** c1
                            q = (a1, b1 + c1)
                            terms[(a, b, c)].append((coef, q, (a - a1, b - b1, c - c1)))
                            qpairs.add(q)
            qpairs = tuple(sorted(qpairs))
            q = np.stack(
                [
                    np.stack(
                        [
                            np.asarray(
                                disc_ridge_function(
                                    np.full(alpha.shape, rho),
                                    alpha,
                                    eq.m_b,
                                    eq.gamma,
                                    dr=i,
                                    da=j,
                                    part=part,
                                )
                            )
                            for part in ("re", "im")
                        ]
                    )
                    for i, j in qpairs
                ]
            )
            self._meta[X] = dict(
                D=tuple(D),
                a2=tuple(a2s),
                b2=tuple(b2s),
                c2=tuple(c2s),
                terms={d: tuple(t) for d, t in terms.items()},
                q=qpairs,
                n_modes=len(key),
            )
            self._constants[X] = dict(idx=idx, rad=rad, g=g, C=C, F=F, q=q)

    def _field(self, X, c):
        meta, K = self._meta[X], self._constants[X]
        cp = jnp.append(c, 0.0)[K["idx"]]  # (m, n, s, slot)
        r = {a2: jnp.sum(cp * K["rad"][i], axis=-1) for i, a2 in enumerate(meta["a2"])}
        T = {}
        for a2 in meta["a2"]:
            for i, c2 in enumerate(meta["c2"]):
                t = jnp.einsum("kn,mns->kms", K["g"][i], r[a2])
                T[a2, c2] = t[..., None] * K["C"][:, :, None, :]  # (k, m, s, u)
        Y = {
            (a2, b2, c2): jnp.einsum(
                "jmu,kmsu->skj", K["F"][meta["b2"].index(b2)], T[a2, c2]
            )
            for a2, b2, c2 in meta["D"]
        }
        qi = {p: i for i, p in enumerate(meta["q"])}
        out = {}
        for d, terms in meta["terms"].items():
            f = Y[d][0]
            for coef, p, d2 in terms:
                q = K["q"][qi[p]]  # (Re/Im, j)
                f = f + coef * (q[0][None, :] * Y[d2][1] + q[1][None, :] * Y[d2][2])
            out[_deriv_key(X, d)] = f.reshape(-1)
        return out

    def compute(self, params):
        """{DESC key: values at the ring nodes} of every R and Z derivative.

        Parameters
        ----------
        params : dict
            Needs "R_lmn" and "Z_lmn".

        Returns
        -------
        data : dict of ndarray
            The ``R_*`` and ``Z_*`` keys on the ring nodes.

        """
        out = {}
        for X in self._derivs:
            out.update(self._field(X, params[X + "_lmn"]))
        return out


def _sqrt_g_on(evaluator, grid, params, keys):
    """Compute keys (√g and its derivatives) on the nodes of a separable evaluator.

    They come from DESC's own formulas, with the evaluator's R, Z data.
    """
    from .utils import _compute as compute_fun

    return compute_fun(
        _EQ,
        keys,
        params=params,
        transforms={"grid": grid},
        profiles={},
        data=evaluator.compute(params),
    )


# ------------------------------------------------------------------------------------
# the ridge subtraction E
# ------------------------------------------------------------------------------------
class RidgeSubtraction(IOAble):
    """The ridge subtraction E of the tied λ, which makes λ constant along the X-lines.

    λ = λ_0 + λ_1 - E with

        E(ρ, θ, ζ) = Σ_k A_k(ρ, α) s_k(ζ),
        A_k = (1/m_b) Σ_{p=0}^{m_b-1} ρ^p cos(p (α - α_k)),

    α = θ - ι_b ζ, α_k = 2πk/m_b and s_k(ζ) the trace of λ_1 on ridge k (ρ = 1,
    α = α_k) minus its mean. At ρ = 1, A_k is 1 on ridge k and 0 on the other ridges,
    and D = ∂_ζ + ι_b ∂_θ annihilates functions of α, so on ridge j: D E = s_j' = D λ_1.
    With λ_0 constant on the X-lines (``FixXLineLambda``), D λ = 0 on every ridge. E is
    a trigonometric polynomial in α with ρ^p factors (regular at the axis),
    NFP-periodic, and odd under stellarator symmetry when λ_1 is.

    When n_b and m_b are coprime, ridge k at ζ is ridge 0 at ζ + 2π q_k/NFP, with
    q_k n_b = k (mod m_b), so every s_k is one function g(u) = λ_1(1, ι_b u, u) of
    period P = 2π m_b/NFP. g is sampled at ``n_zeta`` ζ per field period on every
    ridge (λ_1 from Gauss-Legendre panels of ``n_gl`` points between the corners, where
    √g is smooth) and represented by its trigonometric interpolant without the mean
    and the Nyquist mode (a fixed linear map, so E is linear in the samples and
    differentiable in R, Z; for a stellarator-symmetric equilibrium only the sines are
    kept, g being odd). Its derivatives in ζ are analytic.

    Parameters
    ----------
    eq : RidgeEquilibrium
        The equilibrium.
    n_zeta : int
        Samples per field period and ridge.
    n_gl : int
        Gauss-Legendre points per panel between two corners.

    """

    _static_attrs = ["_shape", "_iota_b", "_m_b"]

    def __init__(self, eq, n_zeta=256, n_gl=32):
        m_b, n_b, NFP = int(eq.m_b), int(eq.n_b), int(eq.NFP)
        errorif(
            gcd(n_b, m_b) != 1,
            NotImplementedError,
            "the ridge subtraction assumes that the m_b ridges form one field line "
            f"(gcd(n_b, m_b) = 1); got n_b = {n_b}, m_b = {m_b}",
        )
        iota_b = eq.iota_b
        zetas = 2 * np.pi / NFP * np.arange(n_zeta) / n_zeta
        x, w = np.polynomial.legendre.leggauss(n_gl)
        h = np.pi / m_b  # half the width of a panel
        aq = (2 * np.pi * np.arange(m_b)[:, None] / m_b + h * (1 + x)[None, :]).ravel()
        th = np.mod(aq[None, :] + iota_b * zetas[:, None], 2 * np.pi).ravel()
        self._grid = Grid(
            np.column_stack([np.ones_like(th), th, np.repeat(zetas, aq.size)]),
            NFP=NFP,
            sort=False,
            jitable=True,
        )
        self._evaluator = SeparableRingEvaluator(
            eq, 1.0, aq, zetas, derivs=_derivs_of(["sqrt(g)"])
        )
        ## the shifts of the ridges along the one field line, and the fit of g
        q = (np.arange(m_b) * pow(n_b, -1, m_b)) % m_b
        self._shift = 2 * np.pi * q / NFP
        P = 2 * np.pi * m_b / NFP
        u = (zetas[:, None] + self._shift[None, :]).ravel()
        J = (u.size - 1) // 2
        self._omega = 2 * np.pi * np.arange(1, J + 1) / P
        ou = self._omega[:, None] * u[None, :]
        fit = np.concatenate([np.cos(ou), np.sin(ou)]) * (2.0 / u.size)
        if eq.sym:
            fit[:J] = 0.0
        self._fit = fit
        self._wq = np.tile(h * w, m_b).reshape(1, -1)
        self._ramp = (2 * np.pi - aq).reshape(1, -1)
        self._iota_b, self._m_b = iota_b, m_b
        self._shape = (n_zeta, m_b, n_gl)
        self._zetas = zetas

    @property
    def zetas(self):
        """ndarray: The planes ζ of the samples of the ridges, on one field period."""
        return self._zetas

    def ridge_lambda_1(self, params):
        """λ_1 = S/G at ρ = 1 on the ridges, shape (n_zeta, m_b).

        The ring of each ζ is the m_b panels between the corners. With b0 the corner
        ridge 0, S(b0) = -(1/2π) ∫ (b0 + 2π - θ) (√g - G) dθ over one turn, and S on
        the next corners adds the integral of √g - G over each panel.
        """
        nz, m_b, ng = self._shape
        sg = _sqrt_g_on(self._evaluator, self._grid, params, ["sqrt(g)"])["sqrt(g)"]
        sg = sg.reshape(nz, m_b * ng)
        G = jnp.sum(self._wq * sg, axis=1) / (2 * np.pi)
        w = sg - G[:, None]
        S0 = -jnp.sum(self._ramp * w * self._wq, axis=1) / (2 * np.pi)
        I = jnp.sum((w * self._wq).reshape(nz, m_b, ng), axis=2)
        Sv = S0[:, None] + jnp.concatenate(
            [jnp.zeros((nz, 1)), jnp.cumsum(I, axis=1)[:, :-1]], axis=1
        )
        return Sv / G[:, None]

    def _phi(self, zetas):
        """The basis of s_k at the ζ: shape (3, n_ζ, m_b, 2J), for d^0, d^1, d^2/dζ."""
        a = self._omega[None, None, :] * (
            np.asarray(zetas)[:, None, None] + self._shift[None, :, None]
        )
        om = self._omega[None, None, :]
        cs, sn = np.cos(a), np.sin(a)
        return np.stack(
            [
                np.concatenate([cs, sn], -1),
                np.concatenate([-om * sn, om * cs], -1),
                np.concatenate([-(om**2) * cs, -(om**2) * sn], -1),
            ]
        )

    def node_constants(self, nodes):
        """The constant arrays of E and its derivatives at the nodes (ρ, θ, ζ)."""
        nodes = np.asarray(nodes, dtype=float)
        m_b = self._m_b
        rho = nodes[:, 0][:, None, None]
        x = (nodes[:, 1] - self._iota_b * nodes[:, 2])[:, None, None] - (
            2 * np.pi * np.arange(m_b) / m_b
        )[None, :, None]
        p = np.arange(m_b)[None, None, :]
        rp = rho**p
        rp1 = np.where(p > 0, p * rho ** np.maximum(p - 1, 0), 0.0)
        cs, sn = np.cos(p * x), np.sin(p * x)
        zs, jz = np.unique(nodes[:, 2], return_inverse=True)
        return dict(
            A=np.sum(rp * cs, -1) / m_b,
            A_t=np.sum(-p * rp * sn, -1) / m_b,
            A_tt=np.sum(-(p**2) * rp * cs, -1) / m_b,
            A_r=np.sum(rp1 * cs, -1) / m_b,
            A_rt=np.sum(-p * rp1 * sn, -1) / m_b,
            phi=self._phi(zs),
            jz=jz.ravel(),
        )

    def compute(self, params, nc):
        """E and its derivatives at the nodes of the node constants ``nc``.

        Returns
        -------
        E : ndarray, shape(num_nodes, 8)
            The columns E, E_t, E_z, E_tt, E_tz, E_zz, E_rt, E_rz.

        """
        coeffs = self._fit @ self.ridge_lambda_1(params).ravel()
        ib = self._iota_b
        s0, s1, s2 = (
            jnp.einsum("zkc,c->zk", nc["phi"][d], coeffs)[nc["jz"]] for d in range(3)
        )
        A, At, Att, Ar, Art = nc["A"], nc["A_t"], nc["A_tt"], nc["A_r"], nc["A_rt"]
        E = [
            A * s0,
            At * s0,
            -ib * At * s0 + A * s1,
            Att * s0,
            -ib * Att * s0 + At * s1,
            ib**2 * Att * s0 - 2 * ib * At * s1 + A * s2,
            Art * s0,
            -ib * Art * s0 + Ar * s1,
        ]
        return jnp.stack([jnp.sum(e, axis=-1) for e in E], axis=-1)


def _derivs_of(keys):
    from .utils import get_derivs

    return get_derivs(keys, _EQ)


# ------------------------------------------------------------------------------------
# the "ring" transform
# ------------------------------------------------------------------------------------
class TiedLambdaRings(IOAble):
    """The ring quadrature of the tied λ on a grid: the ``"ring"`` transform.

    Through every distinct (ρ, ζ) of the grid, a ring sheared with the corners:
    θ_j = α_j + ι_b ζ~, with ζ~ = ζ reduced to [-T/2, T/2) (T = 2π/NFP), so that the
    ring of -ζ is the mirror of the ring of ζ and the tied λ keeps the stellarator
    parity exactly. ``default_ring_rule(ρ, eq.M)`` sets the ring of each radius,
    clustered at the corners (``s_of_a``) outside ρ = 0.7. R and Z on the rings come
    from a ``SeparableRingEvaluator`` per radius; the ring means G and the
    antiderivatives S at the grid's nodes follow ``clustered_ring_matrices``. The ridge
    subtraction E (``RidgeSubtraction``) is evaluated at the grid's nodes too.

    Parameters
    ----------
    eq : RidgeEquilibrium
        The equilibrium.
    grid : Grid
        The grid of the tied λ. It must not have nodes on the axis.

    """

    _static_attrs = ["_shapes"]

    def __init__(self, eq, grid):
        errorif(
            grid.axis.size > 0,
            ValueError,
            "The tied lambda is not defined on the magnetic axis.",
        )
        nodes = np.asarray(grid.nodes)
        iota_b, T = eq.iota_b, 2 * np.pi / eq.NFP
        levels, order, shapes = [], [], []
        for rho in np.unique(nodes[:, 0]):
            sel = np.nonzero(nodes[:, 0] == rho)[0]
            zs = np.unique(nodes[sel, 2])
            zt = zs - T * np.floor(zs / T + 0.5)
            n, eps = default_ring_rule(rho, eq.M)
            jz = np.searchsorted(zs, nodes[sel, 2])
            a, wG, A, rowA = clustered_ring_matrices(
                nodes[sel, 1] - iota_b * zt[jz], n, eps, eq.m_b
            )
            th = np.mod(a[None, :] + iota_b * zt[:, None], 2 * np.pi).ravel()
            ring_grid = Grid(
                np.column_stack([np.full(th.size, rho), th, np.repeat(zs, n)]),
                NFP=eq.NFP,
                sort=False,
                jitable=True,
            )
            levels.append(
                dict(
                    evaluator=SeparableRingEvaluator(eq, rho, a, zs, iota_b * zt),
                    grid=ring_grid,
                    jz=jz,
                    A=A,
                    rowA=rowA,
                    wG=wG,
                )
            )
            order.append(sel)
            shapes.append((zs.size, n))
        self._levels = levels
        self._shapes = tuple(shapes)
        self._inv = np.argsort(np.concatenate(order))
        self._ridge = RidgeSubtraction(eq)
        self._ridge_nodes = self._ridge.node_constants(nodes)

    @property
    def num_ring_nodes(self):
        """int: Number of ring nodes."""
        return int(sum(nz * n for nz, n in self._shapes))

    def ring_data(self, params):
        """The ring means G and antiderivatives S of √g and its derivatives.

        Returns
        -------
        data : ndarray, shape(num_nodes, 10)
            The columns G, G_r, G_z, G_rz, G_zz, S, S_r, S_z, S_rz, S_zz at the grid's
            nodes.

        """
        out = []
        for lv, (nz, n) in zip(self._levels, self._shapes):
            d = _sqrt_g_on(lv["evaluator"], lv["grid"], params, _RING_KEYS)
            f = jnp.stack([d[key].reshape(nz, n) for key in _RING_KEYS])  # (5, nz, n)
            G = (f @ lv["wG"])[:, lv["jz"]]
            S = jnp.einsum("cj,kcj->kc", lv["A"], f[:, lv["jz"]]) - G * lv["rowA"]
            out.append(jnp.concatenate([G, S]).T)
        return jnp.concatenate(out)[self._inv]

    def ridge_data(self, params):
        """E and its derivatives at the grid's nodes (see ``RidgeSubtraction``)."""
        return self._ridge.compute(params, self._ridge_nodes)


# ------------------------------------------------------------------------------------
# the compute functions of the RidgeEquilibrium
# ------------------------------------------------------------------------------------
@register_compute_fun(
    name="tied lambda rings",
    label="(G, G_{\\rho}, G_{\\zeta}, G_{\\rho\\zeta}, G_{\\zeta\\zeta}, S, S_{\\rho}, "
    "S_{\\zeta}, S_{\\rho\\zeta}, S_{\\zeta\\zeta})",
    units="m^{3}",
    units_long="cubic meters",
    description="Ring means G = <sqrt(g)>_theta and zero-mean poloidal antiderivatives "
    "S = A_theta[sqrt(g) - G] of sqrt(g) and of its derivatives r, z, rz, zz, as the "
    "columns G, G_r, G_z, G_rz, G_zz, S, S_r, S_z, S_rz, S_zz",
    dim=10,
    params=["R_lmn", "Z_lmn"],
    transforms={"ring": []},
    profiles=[],
    coordinates="rtz",
    data=[],
    parameterization="desc.equilibrium.equilibrium.RidgeEquilibrium",
)
def _tied_lambda_rings(params, transforms, profiles, data, **kwargs):
    RZ = {"R_lmn": params["R_lmn"], "Z_lmn": params["Z_lmn"]}
    data["tied lambda rings"] = transforms["ring"].ring_data(RZ)
    return data


@register_compute_fun(
    name="E_ridge",
    label="(E, E_{\\theta}, E_{\\zeta}, E_{\\theta\\theta}, E_{\\theta\\zeta}, "
    "E_{\\zeta\\zeta}, E_{\\rho\\theta}, E_{\\rho\\zeta})",
    units="rad",
    units_long="radians",
    description="Ridge subtraction E of the tied stream function and its "
    "derivatives, as the columns E, E_t, E_z, E_tt, E_tz, E_zz, E_rt, E_rz",
    dim=8,
    params=["R_lmn", "Z_lmn"],
    transforms={"ring": []},
    profiles=[],
    coordinates="rtz",
    data=[],
    parameterization="desc.equilibrium.equilibrium.RidgeEquilibrium",
)
def _E_ridge(params, transforms, profiles, data, **kwargs):
    RZ = {"R_lmn": params["R_lmn"], "Z_lmn": params["Z_lmn"]}
    data["E_ridge"] = transforms["ring"].ridge_data(RZ)
    return data


@register_compute_fun(
    name="lambda_1",
    label="\\lambda_1",
    units="rad",
    units_long="radians",
    description="Jacobian-tied part of lambda, S / <sqrt(g)>_theta",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["tied lambda rings"],
    parameterization="desc.equilibrium.equilibrium.RidgeEquilibrium",
)
def _lambda_1(params, transforms, profiles, data, **kwargs):
    G, S = data["tied lambda rings"][:, 0], data["tied lambda rings"][:, 5]
    data["lambda_1"] = S / G
    return data


@register_compute_fun(
    name="lambda_1_t",
    label="\\partial_{\\theta} \\lambda_1",
    units="rad",
    units_long="radians",
    description="Jacobian-tied part of lambda, S / <sqrt(g)>_theta, derivative t",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["sqrt(g)", "tied lambda rings"],
    parameterization="desc.equilibrium.equilibrium.RidgeEquilibrium",
)
def _lambda_1_t(params, transforms, profiles, data, **kwargs):
    G = data["tied lambda rings"][:, 0]
    data["lambda_1_t"] = (data["sqrt(g)"] - G) / G
    return data


@register_compute_fun(
    name="lambda_1_z",
    label="\\partial_{\\zeta} \\lambda_1",
    units="rad",
    units_long="radians",
    description="Jacobian-tied part of lambda, S / <sqrt(g)>_theta, derivative z",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["tied lambda rings"],
    parameterization="desc.equilibrium.equilibrium.RidgeEquilibrium",
)
def _lambda_1_z(params, transforms, profiles, data, **kwargs):
    ring = data["tied lambda rings"]
    G, G_z, S, S_z = ring[:, 0], ring[:, 2], ring[:, 5], ring[:, 7]
    data["lambda_1_z"] = -G_z / G**2 * S + S_z / G
    return data


@register_compute_fun(
    name="lambda_1_tt",
    label="\\partial_{\\theta\\theta} \\lambda_1",
    units="rad",
    units_long="radians",
    description="Jacobian-tied part of lambda, S / <sqrt(g)>_theta, derivative tt",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["sqrt(g)_t", "tied lambda rings"],
    parameterization="desc.equilibrium.equilibrium.RidgeEquilibrium",
)
def _lambda_1_tt(params, transforms, profiles, data, **kwargs):
    G = data["tied lambda rings"][:, 0]
    data["lambda_1_tt"] = data["sqrt(g)_t"] / G
    return data


@register_compute_fun(
    name="lambda_1_tz",
    label="\\partial_{\\theta\\zeta} \\lambda_1",
    units="rad",
    units_long="radians",
    description="Jacobian-tied part of lambda, S / <sqrt(g)>_theta, derivative tz",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["sqrt(g)", "sqrt(g)_z", "tied lambda rings"],
    parameterization="desc.equilibrium.equilibrium.RidgeEquilibrium",
)
def _lambda_1_tz(params, transforms, profiles, data, **kwargs):
    G, G_z = data["tied lambda rings"][:, 0], data["tied lambda rings"][:, 2]
    w, w_z = data["sqrt(g)"] - G, data["sqrt(g)_z"] - G_z
    data["lambda_1_tz"] = -G_z / G**2 * w + w_z / G
    return data


@register_compute_fun(
    name="lambda_1_zz",
    label="\\partial_{\\zeta\\zeta} \\lambda_1",
    units="rad",
    units_long="radians",
    description="Jacobian-tied part of lambda, S / <sqrt(g)>_theta, derivative zz",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["tied lambda rings"],
    parameterization="desc.equilibrium.equilibrium.RidgeEquilibrium",
)
def _lambda_1_zz(params, transforms, profiles, data, **kwargs):
    ring = data["tied lambda rings"]
    G, G_z, G_zz = ring[:, 0], ring[:, 2], ring[:, 4]
    S, S_z, S_zz = ring[:, 5], ring[:, 7], ring[:, 9]
    nu_z, nu_zz = -G_z / G**2, -G_zz / G**2 + 2 * G_z**2 / G**3
    data["lambda_1_zz"] = nu_zz * S + 2 * nu_z * S_z + S_zz / G
    return data


@register_compute_fun(
    name="lambda_1_rt",
    label="\\partial_{\\rho\\theta} \\lambda_1",
    units="rad",
    units_long="radians",
    description="Jacobian-tied part of lambda, S / <sqrt(g)>_theta, derivative rt",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["sqrt(g)", "sqrt(g)_r", "tied lambda rings"],
    parameterization="desc.equilibrium.equilibrium.RidgeEquilibrium",
)
def _lambda_1_rt(params, transforms, profiles, data, **kwargs):
    G, G_r = data["tied lambda rings"][:, 0], data["tied lambda rings"][:, 1]
    w, w_r = data["sqrt(g)"] - G, data["sqrt(g)_r"] - G_r
    data["lambda_1_rt"] = -G_r / G**2 * w + w_r / G
    return data


@register_compute_fun(
    name="lambda_1_rz",
    label="\\partial_{\\rho\\zeta} \\lambda_1",
    units="rad",
    units_long="radians",
    description="Jacobian-tied part of lambda, S / <sqrt(g)>_theta, derivative rz",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["tied lambda rings"],
    parameterization="desc.equilibrium.equilibrium.RidgeEquilibrium",
)
def _lambda_1_rz(params, transforms, profiles, data, **kwargs):
    ring = data["tied lambda rings"]
    G, G_r, G_z, G_rz = ring[:, 0], ring[:, 1], ring[:, 2], ring[:, 3]
    S, S_r, S_z, S_rz = ring[:, 5], ring[:, 6], ring[:, 7], ring[:, 8]
    nu_r, nu_z = -G_r / G**2, -G_z / G**2
    nu_rz = -G_rz / G**2 + 2 * G_r * G_z / G**3
    data["lambda_1_rz"] = nu_rz * S + nu_r * S_z + nu_z * S_r + S_rz / G
    return data


@register_compute_fun(
    name="lambda",
    label="\\lambda",
    units="rad",
    units_long="radians",
    description="Tied stream function lambda_0 + lambda_1 - E",
    dim=1,
    params=["L_lmn"],
    transforms={"L": [[0, 0, 0]]},
    profiles=[],
    coordinates="rtz",
    data=["lambda_1", "E_ridge"],
    parameterization="desc.equilibrium.equilibrium.RidgeEquilibrium",
)
def _tied_lambda(params, transforms, profiles, data, **kwargs):
    lambda_0 = transforms["L"].transform(params["L_lmn"], 0, 0, 0)
    data["lambda"] = lambda_0 + data["lambda_1"] - data["E_ridge"][:, 0]
    return data


@register_compute_fun(
    name="lambda_t",
    label="\\partial_{\\theta} \\lambda",
    units="rad",
    units_long="radians",
    description="Tied stream function lambda_0 + lambda_1 - E, derivative t",
    dim=1,
    params=["L_lmn"],
    transforms={"L": [[0, 1, 0]]},
    profiles=[],
    coordinates="rtz",
    data=["lambda_1_t", "E_ridge"],
    parameterization="desc.equilibrium.equilibrium.RidgeEquilibrium",
)
def _tied_lambda_t(params, transforms, profiles, data, **kwargs):
    lambda_0 = transforms["L"].transform(params["L_lmn"], 0, 1, 0)
    data["lambda_t"] = lambda_0 + data["lambda_1_t"] - data["E_ridge"][:, 1]
    return data


@register_compute_fun(
    name="lambda_z",
    label="\\partial_{\\zeta} \\lambda",
    units="rad",
    units_long="radians",
    description="Tied stream function lambda_0 + lambda_1 - E, derivative z",
    dim=1,
    params=["L_lmn"],
    transforms={"L": [[0, 0, 1]]},
    profiles=[],
    coordinates="rtz",
    data=["lambda_1_z", "E_ridge"],
    parameterization="desc.equilibrium.equilibrium.RidgeEquilibrium",
)
def _tied_lambda_z(params, transforms, profiles, data, **kwargs):
    lambda_0 = transforms["L"].transform(params["L_lmn"], 0, 0, 1)
    data["lambda_z"] = lambda_0 + data["lambda_1_z"] - data["E_ridge"][:, 2]
    return data


@register_compute_fun(
    name="lambda_tt",
    label="\\partial_{\\theta\\theta} \\lambda",
    units="rad",
    units_long="radians",
    description="Tied stream function lambda_0 + lambda_1 - E, derivative tt",
    dim=1,
    params=["L_lmn"],
    transforms={"L": [[0, 2, 0]]},
    profiles=[],
    coordinates="rtz",
    data=["lambda_1_tt", "E_ridge"],
    parameterization="desc.equilibrium.equilibrium.RidgeEquilibrium",
)
def _tied_lambda_tt(params, transforms, profiles, data, **kwargs):
    lambda_0 = transforms["L"].transform(params["L_lmn"], 0, 2, 0)
    data["lambda_tt"] = lambda_0 + data["lambda_1_tt"] - data["E_ridge"][:, 3]
    return data


@register_compute_fun(
    name="lambda_tz",
    label="\\partial_{\\theta\\zeta} \\lambda",
    units="rad",
    units_long="radians",
    description="Tied stream function lambda_0 + lambda_1 - E, derivative tz",
    dim=1,
    params=["L_lmn"],
    transforms={"L": [[0, 1, 1]]},
    profiles=[],
    coordinates="rtz",
    data=["lambda_1_tz", "E_ridge"],
    parameterization="desc.equilibrium.equilibrium.RidgeEquilibrium",
)
def _tied_lambda_tz(params, transforms, profiles, data, **kwargs):
    lambda_0 = transforms["L"].transform(params["L_lmn"], 0, 1, 1)
    data["lambda_tz"] = lambda_0 + data["lambda_1_tz"] - data["E_ridge"][:, 4]
    return data


@register_compute_fun(
    name="lambda_zz",
    label="\\partial_{\\zeta\\zeta} \\lambda",
    units="rad",
    units_long="radians",
    description="Tied stream function lambda_0 + lambda_1 - E, derivative zz",
    dim=1,
    params=["L_lmn"],
    transforms={"L": [[0, 0, 2]]},
    profiles=[],
    coordinates="rtz",
    data=["lambda_1_zz", "E_ridge"],
    parameterization="desc.equilibrium.equilibrium.RidgeEquilibrium",
)
def _tied_lambda_zz(params, transforms, profiles, data, **kwargs):
    lambda_0 = transforms["L"].transform(params["L_lmn"], 0, 0, 2)
    data["lambda_zz"] = lambda_0 + data["lambda_1_zz"] - data["E_ridge"][:, 5]
    return data


@register_compute_fun(
    name="lambda_rt",
    label="\\partial_{\\rho\\theta} \\lambda",
    units="rad",
    units_long="radians",
    description="Tied stream function lambda_0 + lambda_1 - E, derivative rt",
    dim=1,
    params=["L_lmn"],
    transforms={"L": [[1, 1, 0]]},
    profiles=[],
    coordinates="rtz",
    data=["lambda_1_rt", "E_ridge"],
    parameterization="desc.equilibrium.equilibrium.RidgeEquilibrium",
)
def _tied_lambda_rt(params, transforms, profiles, data, **kwargs):
    lambda_0 = transforms["L"].transform(params["L_lmn"], 1, 1, 0)
    data["lambda_rt"] = lambda_0 + data["lambda_1_rt"] - data["E_ridge"][:, 6]
    return data


@register_compute_fun(
    name="lambda_rz",
    label="\\partial_{\\rho\\zeta} \\lambda",
    units="rad",
    units_long="radians",
    description="Tied stream function lambda_0 + lambda_1 - E, derivative rz",
    dim=1,
    params=["L_lmn"],
    transforms={"L": [[1, 0, 1]]},
    profiles=[],
    coordinates="rtz",
    data=["lambda_1_rz", "E_ridge"],
    parameterization="desc.equilibrium.equilibrium.RidgeEquilibrium",
)
def _tied_lambda_rz(params, transforms, profiles, data, **kwargs):
    lambda_0 = transforms["L"].transform(params["L_lmn"], 1, 0, 1)
    data["lambda_rz"] = lambda_0 + data["lambda_1_rz"] - data["E_ridge"][:, 7]
    return data


TIED_LAMBDA_KEYS = [
    "lambda" + s for s in ("", "_t", "_z", "_tt", "_tz", "_zz", "_rt", "_rz")
]
## the other λ keys of an Equilibrium (and their aliases), which the tied λ does not
## provide: RidgeEquilibrium.compute refuses quantities that need them
UNTIED_LAMBDA_KEYS = frozenset(
    key
    for key in data_index[_EQ]
    if key.startswith("lambda_")
    and set(key[len("lambda_") :]) <= set("rtz")
    and not any(
        key == k or key in data_index[_EQ][k]["aliases"] for k in TIED_LAMBDA_KEYS
    )
)
