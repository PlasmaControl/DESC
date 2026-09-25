"""Fitting Fourier toroidal surfaces to points while solving for their angle labels.

The surface is represented as

    R(t, z) = sum_mn R_mn F_mn(t, z)
    Z(t, z) = sum_mn Z_mn F_mn(t, z)
    phi(t, z) = z + sum_mn W_mn F_mn(t, z)

with ``F_mn`` the ``DoubleFourierSeries`` basis. The angles labelling each data point
are unknowns rather than data: any diffeomorphism of the torus that preserves the
winding gives the same surface with different spectral content, so solving for the
labels jointly with the coefficients can represent a surface with far fewer
harmonics than a fit at fixed labels.

The labels are by far the larger block of unknowns (two per point), but each point's
labels enter only that point's residual, so the joint Gauss-Newton system is solved
by eliminating them pointwise with a Schur complement. That leaves a dense system in
the coefficients alone, and a cost per iteration of O(num_points * num_modes**2).

The bases are passed around as a static tuple ``(R_basis, W_basis, Z_basis, w_idx)``,
where ``w_idx`` indexes the omega modes being fit. The coefficients are packed into a
single vector in residual row order ``(R, W[w_idx], Z)``.
"""

import warnings
from functools import partial

import numpy as np

from desc.backend import block_diag, cho_factor, cho_solve, cond, jit, jnp, while_loop
from desc.basis import DoubleFourierSeries
from desc.utils import errorif, warnif

THETA_RULES = ("curvature", "chordal", "centripetal", "uniform")
ZETA_RULES = ("phi", "curvature", "chordal", "centripetal", "uniform")
OMEGA_MODES = ("all", "no_m0", "no_n0", "mixed")

# --------------------------------------------------------------------------
# bases and coefficient packing
# --------------------------------------------------------------------------


def _omega_idx(W_basis, omega_modes):
    """Indices of the omega modes to fit, as a static tuple.

    omega_00 is a rigid toroidal rotation, which is a relabelling rather than a shape,
    so it is always dropped. A relabelling of one angle alone is absorbed by the part
    of omega that depends on that angle alone, so dropping the m=0 and/or n=0 families
    removes that freedom from the model itself.
    """
    errorif(
        omega_modes not in OMEGA_MODES,
        ValueError,
        f"omega_modes should be one of {OMEGA_MODES}, got {omega_modes}",
    )
    m, n = W_basis.modes[:, 1], W_basis.modes[:, 2]
    keep = ~((m == 0) & (n == 0))
    if omega_modes in ("no_m0", "mixed"):
        keep &= m != 0
    if omega_modes in ("no_n0", "mixed"):
        keep &= n != 0
    return tuple(int(i) for i in np.flatnonzero(keep))


def _fourier_bases(M, N, Mw, Nw, NFP, sym, fit_omega, omega_modes="all"):
    """Bases for R, omega and Z, following FourierRZToroidalSurface's conventions."""
    R_basis = DoubleFourierSeries(M=M, N=N, NFP=NFP, sym="cos" if sym else False)
    Z_basis = DoubleFourierSeries(M=M, N=N, NFP=NFP, sym="sin" if sym else False)
    if fit_omega:
        W_basis = DoubleFourierSeries(M=Mw, N=Nw, NFP=NFP, sym="sin" if sym else False)
        w_idx = _omega_idx(W_basis, omega_modes)
    else:
        # empty basis, see FourierRZToroidalSurface.__init__
        W_basis = DoubleFourierSeries(M=0, N=0, NFP=NFP, sym="sin")
        w_idx = ()
    return (R_basis, W_basis, Z_basis, w_idx)


def _w_modes(bases):
    return bases[1].modes[np.array(bases[3], dtype=int)]


def _slices(bases):
    nR, nW, nZ = bases[0].num_modes, len(bases[3]), bases[2].num_modes
    return slice(0, nR), slice(nR, nR + nW), slice(nR + nW, nR + nW + nZ)


def _num_coeffs(bases):
    return bases[0].num_modes + len(bases[3]) + bases[2].num_modes


def _unpack(bases, x):
    """Full R_lmn, W_lmn, Z_lmn from a packed coefficient vector."""
    x = np.asarray(x)
    sR, sW, sZ = _slices(bases)
    W_lmn = np.zeros(bases[1].num_modes)
    W_lmn[np.array(bases[3], dtype=int)] = x[sW]
    return x[sR], W_lmn, x[sZ]


# --------------------------------------------------------------------------
# the surface model
# --------------------------------------------------------------------------


def _nodes(t, z):
    return jnp.stack([jnp.ones_like(t), t, z], axis=-1)


def _model(bases, x, t, z):
    """(R, phi, Z) of the surface at every label pair."""
    R_basis, W_basis, Z_basis, _ = bases
    sR, sW, sZ = _slices(bases)
    nodes = _nodes(t, z)
    return (
        R_basis.evaluate(nodes) @ x[sR],
        z + W_basis.evaluate(nodes, modes=_w_modes(bases)) @ x[sW],
        Z_basis.evaluate(nodes) @ x[sZ],
    )


def _residual(bases, data, x, t, z):
    """Weighted residual against data ``(R, phi, Z, w)``, shape (num_points, 3).

    Taken in cylindrical components scaled to lengths, so that its norm is the true
    Cartesian distance to leading order while staying linear in the coefficients.
    Each point's residual is multiplied by its weight.
    """
    R, p, Z = _model(bases, x, t, z)
    return data[:, 3:] * jnp.stack(
        [R - data[:, 0], data[:, 0] * (p - data[:, 1]), Z - data[:, 2]], axis=-1
    )


def _distance(bases, data, x, t, z):
    """Exact Cartesian distance from each data point to its fitted point."""
    R, p, Z = _model(bases, x, t, z)
    d2 = (
        (R - data[:, 0]) ** 2
        + (Z - data[:, 2]) ** 2
        + 2 * R * data[:, 0] * (1 - jnp.cos(p - data[:, 1]))
    )
    return jnp.sqrt(jnp.maximum(d2, 0.0))


def _cost(bases, data, x, t, z):
    """Weighted sum of squared residuals."""
    return jnp.sum(_residual(bases, data, x, t, z) ** 2)


def _all_derivs(bases, t, z):
    """Bases and their first two label derivatives at every node.

    Returns ``(F, Ft, Fz, Ftt, Ftz, Fzz)`` for each of the R, omega and Z bases, in
    that order. Every one is a product of sin/cos of the mode phase angles, so a
    single phase grid per angle yields all derivative orders: first derivatives shift
    each Fourier factor by pi/2 and scale by its mode number, and second derivatives
    are pure phase shifts and scales of the zeroth.

    Doing it this way makes the whole optimization run 2-3x faster than going through
    the Transform class with a fresh grid at each iteration.
    """
    NFP = bases[0].NFP
    t = jnp.asarray(t)[:, None]
    z = jnp.asarray(z)[:, None]

    def factors(m, n):
        # fourier(x, m, NFP, dx) = (|m|NFP)**dx sin(|m|NFP x + (m>=0)pi/2 + dx pi/2)
        m = jnp.asarray(m, dtype=t.dtype)[None, :]
        n = jnp.asarray(n, dtype=t.dtype)[None, :]
        A = jnp.abs(m) * t + (m >= 0) * (jnp.pi / 2)
        B = NFP * jnp.abs(n) * z + (n >= 0) * (jnp.pi / 2)
        sa, ca = jnp.sin(A), jnp.cos(A)
        sb, cb = jnp.sin(B), jnp.cos(B)
        ma = jnp.abs(m)
        mb = NFP * jnp.abs(n)
        F = sa * sb
        Ft = ma * ca * sb
        Fz = sa * mb * cb
        return F, Ft, Fz, -(ma**2) * F, ma * ca * mb * cb, -(mb**2) * F

    R_modes, Z_modes, W_modes = bases[0].modes, bases[2].modes, _w_modes(bases)
    return (
        factors(R_modes[:, 1], R_modes[:, 2])
        + factors(W_modes[:, 1], W_modes[:, 2])
        + factors(Z_modes[:, 1], Z_modes[:, 2])
    )


# --------------------------------------------------------------------------
# linear fit at fixed labels
# --------------------------------------------------------------------------


def _fit_coeffs(bases, data, t, z):
    """Coefficients minimizing the residual with the labels held fixed.

    The three residual rows are functions of disjoint slices of the coefficient
    vector, so the normal equations are block diagonal and only a num_modes sized
    system is factored.
    """
    R_basis, W_basis, Z_basis, _ = bases
    nodes = _nodes(t, z)
    Rd, pd, Zd, wd = data.T
    C = (
        wd[:, None] * R_basis.evaluate(nodes),
        (wd * Rd)[:, None] * W_basis.evaluate(nodes, modes=_w_modes(bases)),
        wd[:, None] * Z_basis.evaluate(nodes),
    )
    rhs = (wd * Rd, wd * Rd * (pd - z), wd * Zd)
    G = block_diag(*[A.T @ A for A in C])
    b = jnp.concatenate([A.T @ y for A, y in zip(C, rhs)])
    G = G + jnp.diag(1e-14 * jnp.maximum(jnp.diag(G), 1e-30))
    x = cho_solve(cho_factor(G), b)
    # a basis the points cannot resolve leaves G singular, and the Cholesky solve
    # then returns non-finite values; fall back to the minimum norm least squares
    # solution, which is defined either way
    return cond(
        jnp.all(jnp.isfinite(x)),
        lambda: x,
        lambda: jnp.linalg.lstsq(G, b)[0],
    )


# --------------------------------------------------------------------------
# joint Gauss-Newton with the labels eliminated
# --------------------------------------------------------------------------


def _pad_labels(dl):
    """Label step as (num_points, 2), with a zero zeta column if zeta is fixed."""
    if dl.shape[-1] == 2:
        return dl
    return jnp.concatenate([dl, jnp.zeros_like(dl)], axis=-1)


def _lm_state(bases, data, x, t, z):
    """Everything a step needs that does not depend on the damping.

    Returns the coefficient Jacobian blocks ``C``, the label Jacobian ``J``, the
    residual, the damping scales for coefficients and labels, the proxy rms error and
    a cache of second derivatives for the geodesic correction.

    Without omega the toroidal angle is the cylindrical angle, so zeta is fixed by
    the data and only theta is a label unknown: ``J`` then has a single column.
    """
    (
        FR, FtR, FzR, FttR, FtzR, FzzR,
        FW, FtW, FzW, FttW, FtzW, FzzW,
        FZ, FtZ, FzZ, FttZ, FtzZ, FzzZ,
    ) = _all_derivs(bases, t, z)  # fmt: skip
    sR, sW, sZ = _slices(bases)
    Rd, pd, Zd, wd = data.T
    w = wd[:, None]
    xR, xW, xZ = x[sR], x[sW], x[sZ]
    C = (w * FR, (wd * Rd)[:, None] * FW, w * FZ)
    r = jnp.stack(
        [FR @ xR - Rd, Rd * (z + FW @ xW - pd), FZ @ xZ - Zd],
        axis=-1,
    )
    # |r_p| equals the true Cartesian distance to leading order, so the unweighted
    # residual tracks progress for free rather than costing another pass
    e = jnp.sqrt(jnp.mean(jnp.sum(r**2, axis=-1)))
    r = w * r
    # label Jacobian, block diagonal per point: (num_points, row, label). The phi row
    # carries the R weight and the one from d/dzeta of phi = zeta + omega.
    J = jnp.stack(
        [
            jnp.stack([FtR @ xR, Rd * (FtW @ xW), FtZ @ xZ], axis=-1),
            jnp.stack([FzR @ xR, Rd * (1.0 + FzW @ xW), FzZ @ xZ], axis=-1),
        ],
        axis=-1,
    )
    J = w[..., None] * J
    if not len(bases[3]):
        J = J[..., :1]
    # Damping is scaled by the square root of the Gauss-Newton Hessian diagonal, so
    # low curvature directions keep longer steps than high curvature ones, but by
    # less than plain Marquardt scaling would give them. This sits between pure
    # Levenberg and Marquardt damping and is the most consistently robust choice.
    Dx = jnp.concatenate([jnp.einsum("pi,pi->i", A, A) for A in C])
    Dl = jnp.einsum("pki,pki->pi", J, J)
    Dx = jnp.maximum(Dx, 1e-30) ** 0.5
    Dl = jnp.maximum(Dl, 1e-30) ** 0.5
    # second derivatives contracted with the current coefficients, and first
    # derivatives left raw for the cross terms with the per trial coefficient step
    Gtt = jnp.stack([FttR @ xR, Rd * (FttW @ xW), FttZ @ xZ], axis=-1)
    Gtz = jnp.stack([FtzR @ xR, Rd * (FtzW @ xW), FtzZ @ xZ], axis=-1)
    Gzz = jnp.stack([FzzR @ xR, Rd * (FzzW @ xW), FzzZ @ xZ], axis=-1)
    K = (Gtt, Gtz, Gzz, FtR, FzR, FtW, FzW, FtZ, FzZ)
    return C, J, r, Dx, Dl, e, K


def _lm_operator(state, lam):
    """Factors of the damped reduced system, shared by every right hand side.

    Each point's labels appear only in that point's residual, so the label block of
    the Gauss-Newton Hessian inverts pointwise. Eliminating it leaves a dense system
    in the coefficients alone. ``V`` projects each point's residual onto the
    direction that sliding its labels along the surface cannot fix (the surface
    normal, when undamped), so only that component drives the coefficients.
    """
    C, J, _, Dx, Dl = state[:5]
    k = J.shape[-1]
    H = jnp.einsum("pki,pkj->pij", J, J)
    H = H + (lam * Dl)[:, None, :] * jnp.eye(k)
    Hi = jnp.linalg.inv(H)
    V = jnp.eye(3) - jnp.einsum("pik,pkl,pjl->pij", J, Hi, J)

    # S is symmetric because V is, so only the upper blocks are formed
    blocks = [[None] * 3 for _ in range(3)]
    for a in range(3):
        for b in range(a, 3):
            Sab = (C[a] * V[:, a, b, None]).T @ C[b]
            blocks[a][b] = Sab
            if b != a:
                blocks[b][a] = Sab.T
    # SPD by construction: a Gauss-Newton Hessian plus positive damping. If it is
    # not, the solve returns NaN, the trial is rejected on cost, and the damping goes
    # up, which is the right response anyway.
    S = jnp.block(blocks) + jnp.diag(lam * Dx)
    return Hi, cho_factor(S)[0]


def _lm_solve(bases, state, op, rr):
    """Coefficient and label steps of the reduced system for right hand side ``rr``."""
    C, J = state[0], state[1]
    Hi, chol = op
    g = jnp.einsum("pki,pk->pi", J, rr)
    rq = rr - jnp.einsum("pik,pkl,pl->pi", J, Hi, g)
    gs = jnp.concatenate([C[a].T @ rq[:, a] for a in range(3)])
    dx = cho_solve((chol, False), -gs)
    slices = _slices(bases)
    Adx = jnp.stack([C[a] @ dx[slices[a]] for a in range(3)], axis=-1)
    dl = -jnp.einsum("pij,pj->pi", Hi, g + jnp.einsum("pki,pk->pi", J, Adx))
    return dx, dl


def _d2(bases, data, K, dx, dl):
    """Second derivative of the residual along a step in the joint space.

    The residual is linear in the coefficients, so the only terms are second
    derivatives in the labels and cross terms between labels and coefficients.
    """
    Gtt, Gtz, Gzz, FtR, FzR, FtW, FzW, FtZ, FzZ = K
    sR, sW, sZ = _slices(bases)
    Rd = data[:, 0]
    dl = _pad_labels(dl)
    dt, dz = dl[:, 0:1], dl[:, 1:2]
    Vt = jnp.stack([FtR @ dx[sR], Rd * (FtW @ dx[sW]), FtZ @ dx[sZ]], axis=-1)
    Vz = jnp.stack([FzR @ dx[sR], Rd * (FzW @ dx[sW]), FzZ @ dx[sZ]], axis=-1)
    return data[:, 3:] * (
        Gtt * dt**2 + 2 * Gtz * dt * dz + Gzz * dz**2 + 2 * (Vt * dt + Vz * dz)
    )


def _model_cost(bases, state, dx, dl):
    """Cost the linearization at the current point predicts for a step."""
    C, J, r = state[0], state[1], state[2]
    slices = _slices(bases)
    rl = (
        r
        + jnp.stack([C[a] @ dx[slices[a]] for a in range(3)], axis=-1)
        + jnp.einsum("pki,pi->pk", J, dl)
    )
    return jnp.sum(rl**2)


# Largest geodesic correction accepted, relative to the step it corrects. A
# correction approaching the size of that step means the quadratic picture does not
# reach as far as the step is trying to go.
_ACCEL_ALPHA = 0.75


def _joint_solve(bases, data, x, t, z, maxiter, ftol, err_target):
    """Levenberg-Marquardt on labels and coefficients together.

    Nothing fixes the gauge: a reparameterization leaves the surface where it is, so
    the labels are determined only up to one, and the iteration settles on whichever
    member of the family it reaches. Only the surface is asked for, so that costs
    nothing. As the damping grows the eliminated label block contributes less and the
    step tends to the fixed label linear fit, so the damping alone globalizes the
    iteration.

    The eliminated problem is flat along a reparameterization, which changes the
    labels and the coefficients together while barely moving the surface. A straight
    step leaves that valley almost at once, so each step gets a geodesic correction
    from the second derivative of the residual along it, which curves it to follow
    the valley. The correction solves against the factors the step was built from,
    so it costs a back substitution rather than a second assembly.

    Iteration stops when the unweighted rms distance reaches ``err_target``, when
    the cost stops falling by ``ftol`` relative per iteration, or at ``maxiter``. A
    step is only accepted if it lowers the cost, so the last iterate is always the
    best one visited.
    """

    def trials(state, x, t, z, F, lam, nu):
        """Raise the damping until a trial step lowers the cost, or give up."""

        def cond_fun(c):
            return (c[0] < 12) & jnp.logical_not(c[3])

        def body(c):
            k, lam, nu, _, xc, tc, zc, Fc = c
            op = _lm_operator(state, lam)
            dx, dl = _lm_solve(bases, state, op, state[2])
            # geodesic correction, solved against the factors just built
            ax, al = _lm_solve(bases, state, op, _d2(bases, data, state[6], dx, dl))
            # compared as squares, so that a step shrinking to nothing near the
            # solution leaves the test differentiable
            nv2 = jnp.sum(dx**2) + jnp.sum(dl**2)
            na2 = jnp.sum(ax**2) + jnp.sum(al**2)
            use = jnp.where(na2 <= _ACCEL_ALPHA**2 * nv2, 0.5, 0.0)
            dx, dl = dx + use * ax, dl + use * al
            dl2 = _pad_labels(dl)
            tn, zn, xn = t + dl2[:, 0], z + dl2[:, 1], x + dx
            Fn = _cost(bases, data, xn, tn, zn)
            acc = Fn < F
            # share of the predicted improvement the step delivered. Near one the
            # linearization is holding and the damping can be relaxed; well below it
            # the step went further than the model can vouch for. The floor bounds
            # how fast the damping may fall.
            pred = F - _model_cost(bases, state, dx, dl)
            rho = (F - Fn) / jnp.maximum(pred, jnp.finfo(F.dtype).tiny)
            # a rejected trial raises the damping by nu, and nu itself grows on each
            # consecutive rejection so the next attempt lands on a gentler step
            lam = jnp.where(
                acc,
                jnp.maximum(
                    lam * jnp.maximum(0.7, 1.0 - (2.0 * rho - 1.0) ** 3), 1e-14
                ),
                jnp.minimum(lam * nu, 1e14),
            )
            nu = jnp.where(acc, 3.0, jnp.minimum(nu * 3.0, 1e10))

            def keep(new, old):
                return jnp.where(acc, new, old)

            return (
                k + 1, lam, nu, acc,
                keep(xn, xc), keep(tn, tc), keep(zn, zc), keep(Fn, Fc),
            )  # fmt: skip

        return while_loop(cond_fun, body, (0, lam, nu, jnp.asarray(False), x, t, z, F))

    def outer_cond(c):
        return (c[0] < maxiter) & jnp.logical_not(c[-1])

    def outer_body(c):
        nit, x, t, z, F, lam, nu, _ = c
        state = _lm_state(bases, data, x, t, z)
        e = state[5]
        # the proxy is only a leading order distance, so confirm it against the
        # exact one before stopping, and only when it claims success
        hit = cond(
            e <= err_target,
            lambda: jnp.sqrt(jnp.mean(_distance(bases, data, x, t, z) ** 2))
            <= err_target,
            lambda: jnp.asarray(False),
        )
        _, lam_n, nu_n, acc, xn, tn, zn, Fn = cond(
            hit,
            lambda: (0, lam, nu, jnp.asarray(False), x, t, z, F),
            lambda: trials(state, x, t, z, F, lam, nu),
        )
        done = hit | jnp.logical_not(acc) | (F - Fn <= ftol * F)
        return (nit + 1, xn, tn, zn, Fn, lam_n, nu_n, done)

    init = (
        0, x, t, z, _cost(bases, data, x, t, z), jnp.asarray(1e-3), jnp.asarray(3.0),
        jnp.asarray(False),
    )  # fmt: skip
    nit, x, t, z, _, _, _, _ = while_loop(outer_cond, outer_body, init)
    return x, t, z, nit


@partial(jit, static_argnames=["bases"])
def _optimize_labels(bases, data, t0, z0, maxiter, ftol, err_target):
    """Coefficients and labels at one resolution, starting from the given labels.

    Parameters
    ----------
    bases : tuple
        ``(R_basis, W_basis, Z_basis, w_idx)``.
    data : ndarray, shape(num_points, 4)
        Points in cylindrical (R, phi, Z), with phi on the same branch as ``z0``, and
        the weight of each point's residual.
    t0, z0 : ndarray, shape(num_points,)
        Initial labels. If no omega modes are fit, ``z0`` must equal phi and is held
        fixed.
    maxiter : int
        Maximum joint iterations. With 0 only the linear fit at the initial labels is
        done.
    ftol : float
        Relative decrease of the cost below which iteration stops.
    err_target : float
        Unweighted rms distance at which iteration stops.

    Returns
    -------
    x : ndarray
        Packed coefficients.
    t, z : ndarray
        Labels.
    nit : int
        Number of joint iterations taken.

    """
    x = _fit_coeffs(bases, data, t0, z0)
    return _joint_solve(bases, data, x, t0, z0, maxiter, ftol, err_target)


def _prepare_data(coords, zeta, w=None):
    """Data array (R, phi, Z, w) with phi moved onto the branch of the given zeta."""
    R, phi, Z = coords[..., 0].ravel(), coords[..., 1].ravel(), coords[..., 2].ravel()
    zeta = np.ravel(zeta)
    w = np.ones_like(R) if w is None else np.ravel(w)
    # exact while |omega| < pi, whatever 2 pi branch either input is on
    phi = zeta + np.arctan2(np.sin(phi - zeta), np.cos(phi - zeta))
    return jnp.asarray(np.stack([R, phi, Z, w], axis=-1))


def _check_determined(bases, num_points):
    """Warn if the joint fit has more unknowns than it has data."""
    # solving for the labels spends one (fixed zeta) or two of the three residuals
    # per point on them, leaving the rest for all the coefficients together
    num_labels = 1 if not len(bases[3]) else 2
    n = _num_coeffs(bases)
    warnif(
        n > (3 - num_labels) * num_points,
        UserWarning,
        f"Fitting {n} coefficients and {num_labels} label(s) per point to "
        f"{num_points} points is underdetermined.",
    )


# --------------------------------------------------------------------------
# initial labels
# --------------------------------------------------------------------------


def _neighbours(pts, turn=0.0):
    """Points before and after each point of a closed curve, (..., n, 3).

    ``turn`` is the rotation about the z axis that carries the curve onto its own
    continuation, so that a curve spanning whole field periods closes into the next
    period rather than back onto its own start. It is zero for a curve that closes
    on itself, such as a cross section.
    """
    c, s = jnp.cos(turn), jnp.sin(turn)
    rot = jnp.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    nxt = jnp.concatenate([pts[..., 1:, :], pts[..., :1, :] @ rot.T], axis=-2)
    prv = jnp.concatenate([pts[..., -1:, :] @ rot, pts[..., :-1, :]], axis=-2)
    return prv, nxt


def _menger_curvature(prv, pts, nxt):
    """Discrete curvature of 3D polygons from the circle through consecutive points."""
    a, b = prv - pts, nxt - pts
    c = nxt - prv
    la, lb, lc = (jnp.linalg.norm(v, axis=-1) for v in (a, b, c))
    area = 0.5 * jnp.linalg.norm(jnp.cross(a, b), axis=-1)
    denom = la * lb * lc
    return jnp.where(denom > 0, 4 * area / jnp.where(denom > 0, denom, 1.0), 0.0)


def _centroids(xyz):
    """Centroid of each cross section, (nt, nz, 3) -> (nz, 3).

    Each point carries the poloidal arclength it represents, so the curve of
    centroids follows the shape of the cross sections rather than how densely any
    part of them happens to be sampled.
    """
    seg = jnp.linalg.norm(jnp.roll(xyz, -1, axis=0) - xyz, axis=-1)
    w = 0.5 * (seg + jnp.roll(seg, 1, axis=0))
    return jnp.sum(w[..., None] * xyz, axis=0) / jnp.sum(w, axis=0)[:, None]


def _toroidal_span(phi_un, NFP):
    """Toroidal extent of the mesh, as a whole number of field periods.

    A closed surface is sampled over whole field periods, so the extent estimated
    from the sampled angles is rounded to the nearest multiple of 2 pi / NFP, which
    makes it exact even where the cylindrical angle is sampled unevenly.
    """
    nz = phi_un.shape[1]
    est = jnp.mean(phi_un[:, -1] - phi_un[:, 0]) * nz / max(nz - 1, 1)
    turns = jnp.maximum(jnp.round(est * NFP / (2 * jnp.pi)), 1.0)
    return turns * 2 * jnp.pi / NFP


def _spacing(pts, method, turn=0.0, exponent=1.0 / 3.0, floor=0.05):
    """Relative label increments along closed curves, (..., n, 3) -> (..., n).

    For ``method="curvature"`` the increment is ``kappa**(1/3) ds``. This is exact
    for an ellipse: it recovers the single harmonic parameterization
    ``(a cos(t), b sin(t))`` from any sampling of it, which equal arclength spacing
    does not.
    """
    prv, nxt = _neighbours(pts, turn)
    seg = jnp.linalg.norm(nxt - pts, axis=-1)  # i -> i+1
    if method == "uniform":
        w = jnp.ones_like(seg)
    elif method == "chordal":
        w = seg
    elif method == "centripetal":
        w = jnp.sqrt(seg)
    else:
        kap = _menger_curvature(prv, pts, nxt)
        # the floor keeps nearly straight segments from collapsing to zero weight;
        # it is set from each curve's own mean, so curves of different size spaced
        # together are treated alike
        kap = kap + floor * jnp.mean(kap, axis=-1, keepdims=True)
        w = seg * (0.5 * (kap + jnp.roll(kap, -1, axis=-1))) ** exponent
    return jnp.where(w > 0, w, jnp.finfo(seg.dtype).tiny)


def _cumulative(w, period):
    """Monotone parameter in [0, period) from increments along the last axis."""
    c = jnp.cumsum(w, axis=-1)
    c = jnp.concatenate([jnp.zeros_like(c[..., :1]), c[..., :-1]], axis=-1)
    return period * c / jnp.sum(w, axis=-1, keepdims=True)


def _signed_area(R, Z):
    """Shoelace area of closed polygons (R, Z) along the last axis, + for CCW."""
    return 0.5 * jnp.sum(
        R * jnp.roll(Z, -1, axis=-1) - jnp.roll(R, -1, axis=-1) * Z, axis=-1
    )


def initial_labels(coords, NFP=1, sym=False, theta="curvature", zeta="curvature"):
    """Construct initial (theta, zeta) labels for a structured mesh of points.

    Parameters
    ----------
    coords : ndarray, shape(nt, nz, 3)
        Points in cylindrical (R, phi, Z). The first axis runs poloidally around the
        torus, the second toroidally over a whole number of field periods.
    NFP : int
        Number of field periods.
    sym : bool
        Whether the surface is stellarator symmetric. If True the labels are anchored
        so that theta = zeta = 0 stays at the mesh origin, preserving the phase the
        symmetry is expressed in. Otherwise theta is anchored to the geometric
        poloidal angle about each cross section's centroid, and zeta is offset so
        that omega starts with zero mean.
    theta : {"curvature", "chordal", "centripetal", "uniform"}
        Rule for spacing the poloidal labels around each cross section.
    zeta : {"phi", "curvature", "chordal", "centripetal", "uniform"}
        Rule for the toroidal labels. ``"phi"`` uses the cylindrical angle itself, so
        omega starts at zero. The others space zeta by that rule along the curve of
        cross section centroids, following the shape of the surface rather than the
        angle it is viewed from.

    Returns
    -------
    theta, zeta : ndarray, shape(nt, nz)
        Initial angle labels.

    """
    errorif(
        theta not in THETA_RULES,
        ValueError,
        f"theta rule should be one of {THETA_RULES}, got {theta}",
    )
    errorif(
        zeta not in ZETA_RULES,
        ValueError,
        f"zeta rule should be one of {ZETA_RULES}, got {zeta}",
    )
    coords = jnp.asarray(coords)
    nt, nz, _ = coords.shape
    R, phi, Z = coords[..., 0], coords[..., 1], coords[..., 2]
    xyz = jnp.stack([R * jnp.cos(phi), R * jnp.sin(phi), Z], axis=-1)
    phi_un = jnp.unwrap(phi, axis=1)

    if zeta == "phi":
        zeta_out = phi_un
    else:
        # The labels are spaced along the curve of cross section centroids, which is
        # where an uneven advance of the cylindrical angle costs toroidal harmonics.
        # One profile is shared by all poloidal indices, so that surfaces of constant
        # zeta stay close to the mesh's cross sections.
        period = _toroidal_span(phi_un, NFP)
        z1 = _cumulative(_spacing(_centroids(xyz), zeta, turn=period), period)
        if not sym:
            z1 = z1 + (phi_un.mean() - z1.mean())
        zeta_out = jnp.broadcast_to(z1, (nt, nz))

    # the cross sections are spaced together, as a stack of closed curves with the
    # poloidal index last
    raw = _cumulative(_spacing(jnp.swapaxes(xyz, 0, 1), theta), 2 * jnp.pi)
    if sym:
        # theta = 0 on the mesh seam keeps the surface's own symmetric phase, which
        # anchoring to a geometric angle would rotate off the cos/sin basis
        return np.asarray(raw.T), np.asarray(zeta_out)
    Rj, Zj = R.T, Z.T
    # anchor each cross section against the geometric poloidal angle about its
    # centroid, so that the theta origin varies smoothly with zeta
    alpha = jnp.arctan2(
        Zj - Zj.mean(-1, keepdims=True), Rj - Rj.mean(-1, keepdims=True)
    )
    # where the data runs clockwise in (R, Z) that angle runs backwards
    alpha = jnp.where(_signed_area(Rj, Zj)[:, None] < 0, -alpha, alpha)
    d = alpha - raw
    off = jnp.arctan2(jnp.mean(jnp.sin(d), axis=-1), jnp.mean(jnp.cos(d), axis=-1))
    theta_out = (raw + off[:, None]).T
    # remove any 2 pi jumps in the per cross section offset
    theta_out -= 2 * jnp.pi * jnp.round((theta_out - theta_out[:, :1]) / (2 * jnp.pi))
    return np.asarray(theta_out), np.asarray(zeta_out)


# --------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------


def _label_spacing(theta, zeta):
    """Smallest label increment between adjacent points of a mesh, in each angle.

    Signed so that it is positive whenever the labels are ordered, whichever way
    round the mesh they run: a negative value means neighbouring points have crossed.
    """
    st = np.sign(np.median(np.diff(theta, axis=0))) or 1.0
    dt = st * np.diff(np.vstack([theta, theta[:1] + st * 2 * np.pi]), axis=0)
    if zeta.shape[1] < 2:
        return float(dt.min()), np.inf
    sz = np.sign(np.median(np.diff(zeta, axis=1))) or 1.0
    return float(dt.min()), float((sz * np.diff(zeta, axis=1)).min())


def _check_label_order(theta, zeta):
    """Warn if neighbouring points of a mesh have crossed in either label."""
    min_dt, min_dz = _label_spacing(theta, zeta)
    if min_dt <= 0 or min_dz <= 0:
        warnings.warn(
            "Optimized labels are not monotonic along the mesh (min theta spacing "
            f"{min_dt:.3e}, min zeta spacing {min_dz:.3e}), so the fitted "
            "parameterization folds over."
        )


def _report(bases, rms_err, max_err, nit, scale=None, tol=None):
    """Print a summary of a fit."""
    R_basis, W_basis, _, w_idx = bases

    def rel(e):
        return f"  ({e / scale:.3e} of mean minor radius)" if scale else ""

    print(f"Optimized surface fit at M={R_basis.M}, N={R_basis.N}", end="")
    print(f", omega Mw={W_basis.M}, Nw={W_basis.N}" if len(w_idx) else ", omega = 0")
    print(f"  number of coefficients : {_num_coeffs(bases)}")
    print(f"  rms distance to points : {rms_err:.4e}" + rel(rms_err))
    print(f"  max distance to points : {max_err:.4e}" + rel(max_err))
    if tol is not None:
        print(f"  relative rms tolerance : {tol:.3e}")
    print(f"  iterations of last fit : {nit}")


def _mean_minor_radius(coords):
    """Mean distance from each cross section's centroid to its points."""
    Rc = coords[..., 0].mean(axis=0, keepdims=True)
    Zc = coords[..., 2].mean(axis=0, keepdims=True)
    return float(np.mean(np.hypot(coords[..., 0] - Rc, coords[..., 2] - Zc)))


# --------------------------------------------------------------------------
# resolution search
# --------------------------------------------------------------------------


def _num_basis(M, N):
    return (2 * M + 1) * (2 * N + 1)


def _spectral_estimate(data, t0, z0, M_min, N_min, M_max, N_max, basis_kwargs, tol):
    """Cheapest truncation the fixed label spectrum says can meet the tolerance.

    A linear fit at the maximum resolution gives the full (m, n) spectrum of the
    surface at the initial labels. The truncation error a resolution (M, N) would
    leave is the energy of every mode with |m| > M or |n| > N. The labels the joint
    solve converges to are different, so this sets a direction to search along, not
    the answer itself.
    """
    bases = _fourier_bases(M_max, N_max, M_max, N_max, **basis_kwargs)
    x, _, _, _ = _optimize_labels(bases, data, t0, z0, 0, np.inf, 0.0)
    x = np.asarray(x)
    sR, sW, sZ = _slices(bases)
    rd = float(np.mean(data[:, 0]))
    energy = {}
    for modes, c in (
        (bases[0].modes, x[sR]),
        (_w_modes(bases), x[sW] * rd),
        (bases[2].modes, x[sZ]),
    ):
        for (m, n), ci in zip(modes[:, 1:], c):
            energy[(m, n)] = energy.get((m, n), 0.0) + ci * ci
    mn = np.array(list(energy.keys())).reshape(-1, 2)
    E = np.array(list(energy.values()))

    best, best_cost = (M_max, N_max), np.inf
    for m in range(M_min, M_max + 1):
        for n in range(N_min, N_max + 1):
            tail = np.sqrt(E[(np.abs(mn[:, 0]) > m) | (np.abs(mn[:, 1]) > n)].sum())
            if _num_basis(m, n) < best_cost and tail <= tol:
                best, best_cost = (m, n), _num_basis(m, n)
    return best


def _condense(  # noqa: C901
    coords,
    theta0,
    zeta0,
    tol,
    M_min,
    N_min,
    M_max,
    N_max,
    basis_kwargs,
    maxiter,
    ftol,
    verbose,
):
    """Search for the cheapest resolution whose optimized fit meets ``tol``.

    Cost is the number of basis functions (2M+1)(2N+1). Acceptance is on the rms
    distance, which is what the solve minimizes and so falls monotonically with it,
    unlike the max distance.

    The search climbs from (M_min, N_min) along the direction estimated from the
    fixed label spectrum, at geometrically growing fractions of the way to the
    bound, bisects between the last failure and the first success, then moves to
    the cheapest passing neighbor in the surrounding 3x3 block until none is
    cheaper. Each fit warm starts its labels from the best passing fit so far.

    Returns
    -------
    best : dict
        Keys ``M, N, bases, x, theta, zeta, rms_err, max_err, nit, ok``.
    history : list of dict
        Every resolution tried, with its errors.

    """
    nt, nz, _ = coords.shape
    data = _prepare_data(coords, zeta0)
    t0 = jnp.asarray(np.ravel(theta0))
    z0 = jnp.asarray(np.ravel(zeta0))
    bases = _fourier_bases(M_max, N_max, M_max, N_max, **basis_kwargs)
    _check_determined(bases, nt * nz)

    history = []
    cache = {}
    warm = [t0, z0]

    def attempt(M, N):
        if (M, N) in cache:
            return cache[(M, N)]
        bases = _fourier_bases(M, N, M, N, **basis_kwargs)
        x, t, z, nit = _optimize_labels(
            bases, data, warm[0], warm[1], maxiter, ftol, tol
        )
        err = np.asarray(_distance(bases, data, x, t, z))
        res = dict(
            M=M, N=N, bases=bases, x=x, theta=t, zeta=z, nit=int(nit),
            rms_err=float(np.sqrt(np.mean(err**2))), max_err=float(err.max()),
        )  # fmt: skip
        res["ok"] = res["rms_err"] <= tol
        cache[(M, N)] = res
        history.append({k: res[k] for k in ("M", "N", "rms_err", "max_err", "ok")})
        if verbose > 1:
            print(
                f"  M={M:2d} N={N:2d}  rms err={res['rms_err']:.3e}  "
                f"max err={res['max_err']:.3e}  "
                f"{'accept' if res['ok'] else 'reject'}"
            )
        return res

    M_est, N_est = _spectral_estimate(
        data, t0, z0, M_min, N_min, M_max, N_max, basis_kwargs, tol
    )
    dM, dN = M_est - M_min, N_est - N_min
    if dM and dN:
        # keep the ratio of the estimated direction, scaled to reach the bound
        s = min((M_max - M_min) / dM, (N_max - N_min) / dN)
        dM, dN = round(dM * s), round(dN * s)
        if dM == 0 and dN == 0:
            dM, dN = M_max - M_min, N_max - N_min
    elif dM:
        dM = M_max - M_min
    elif dN:
        dN = N_max - N_min
    else:
        dM, dN = M_max - M_min, N_max - N_min

    def ray(s):
        return int(M_min + round(s * dM)), int(N_min + round(s * dN))

    lo, hi, best = 0.0, 1.0, None
    for frac in (0.0, 0.0625, 0.125, 0.25, 0.5, 1.0):
        res = attempt(*ray(frac))
        if res["ok"]:
            hi, best = frac, res
            break
        lo = frac
    if best is None:
        # nothing along the search direction passes, so fall back to the bound
        best = attempt(M_max, N_max)
    else:
        warm = [best["theta"], best["zeta"]]
        while ray(lo) != ray(hi):
            mid = 0.5 * (lo + hi)
            if ray(mid) in (ray(lo), ray(hi)):
                break
            res = attempt(*ray(mid))
            if res["ok"]:
                hi, best = mid, res
                warm = [res["theta"], res["zeta"]]
            else:
                lo = mid
    # A move like (M+1, N-1) can be cheaper than (M, N) even though it raises one
    # index, so this considers the whole neighbourhood rather than only decreases.
    while best["ok"]:
        m0, n0 = best["M"], best["N"]
        moves = sorted(
            (_num_basis(m, n), m, n)
            for m in (m0 - 1, m0, m0 + 1)
            for n in (n0 - 1, n0, n0 + 1)
            if (m, n) != (m0, n0) and M_min <= m <= M_max and N_min <= n <= N_max
        )
        next_best = None
        for cost, m, n in moves:
            if cost >= _num_basis(m0, n0):
                break
            res = attempt(m, n)
            if res["ok"]:
                next_best = res
                break
        if next_best is None:
            break
        best = next_best
        warm = [best["theta"], best["zeta"]]

    best["theta"] = np.asarray(best["theta"]).reshape(nt, nz)
    best["zeta"] = np.asarray(best["zeta"]).reshape(nt, nz)
    return best, history
