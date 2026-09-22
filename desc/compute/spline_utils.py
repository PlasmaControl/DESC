"""Helper functions for curve and surface classes."""

from desc.backend import jnp


def uniform_knots(n, p, domain=2 * jnp.pi):
    """Periodic, uniformly-spaced knot vector for a degree-p B-spline.

    Parameters
    ----------
    n : int
        Control point count minus 1 (closed loop of n+1 points).
    p : int
        Degree.
    domain : float
        Period of the curve.

    Returns
    -------
    knots : ndarray
        Knot vector, padded by p points on each side to match a control
        array tiled the same way (length n+3p+2).
    """
    interval = domain / (n + 1)
    k = jnp.arange(-2 * p, n + 2 * p + 2)
    knots_wide = k * interval
    n_needed = (n + 1 + 2 * p) + p + 1
    start = (len(knots_wide) - n_needed) // 2
    return knots_wide[start : start + n_needed]


def chord_length_knots(points, p, domain=2 * jnp.pi):
    """Periodic, chord-length-parametrized knot vector for a degree-p B-spline.

    Parameters
    ----------
    points : ndarray, shape(n+1, dim)
        Control points for one period (no tiled copies).
    p : int
        Degree.
    domain : float
        Period of the curve.

    Returns
    -------
    knots : ndarray
        Knot vector spaced by Euclidean distance between points (wrapping
        the last point back to the first), padded the same way as
        `uniform_knots` (length n+3p+2).
    """
    n = len(points) - 1
    closed = jnp.concatenate([points, points[:1]], axis=0)  # (n+2, dim): +1 closing gap
    gaps = jnp.linalg.norm(jnp.diff(closed, axis=0), axis=1)  # (n+1,)
    total = jnp.sum(gaps)
    if total <= 0:
        gaps = jnp.ones(n + 1)
        total = n + 1

    t_core = jnp.concatenate([[0.0], jnp.cumsum(gaps)]) * (domain / total)

    k = jnp.arange(-2 * p, n + 2 * p + 2)
    m = k % (n + 1)
    shift = (k - m) // (n + 1)
    knots_wide = t_core[m] + shift * domain

    n_needed = (n + 1 + 2 * p) + p + 1
    start = (len(knots_wide) - n_needed) // 2
    return knots_wide[start : start + n_needed]


def b_p_deriv3(s, p, t):
    """B-spline basis functions and their 1st, 2nd, and 3rd derivatives.

    Parameters
    ----------
    s : ndarray
        Points to evaluate at.
    p : int
        Degree.
    t : ndarray
        Knot vector.

    Returns
    -------
    b, b_s, b_ss, b_sss : ndarray, shape(len(s), n_basis)
        Basis functions and their successive derivatives w.r.t. s.
    """

    def _safe_divide(num, den):
        return jnp.where(den != 0, num / jnp.where(den != 0, den, 1), 0)

    b = []
    tt = t
    ss = s
    for deg in range(0, p + 1):
        if deg == 0:
            x1d = ss.copy()
            t1d = tt.copy()
            tt = jnp.outer(jnp.ones(len(ss)), tt)
            ss = jnp.expand_dims(ss, 0)
            b0 = ((ss.T >= tt[:, :-1]) & (ss.T < tt[:, 1:])).astype(tt.dtype)
            b0 = b0.at[jnp.isclose(x1d, t1d[-1]), jnp.isclose(t1d[1:], t1d[-1])].set(
                1.0
            )
            b.append(b0)
        else:
            l_term_n = ss.T - tt[:, : -deg - 1]
            l_term_d = tt[:, deg:-1] - tt[:, : -deg - 1]
            l_term = b[-1][:, :-1] * _safe_divide(l_term_n, l_term_d)

            r_term_n = tt[:, deg + 1 :] - ss.T
            r_term_d = tt[:, deg + 1 :] - tt[:, 1:-deg]
            r_term = b[-1][:, 1:] * _safe_divide(r_term_n, r_term_d)

            b.append(l_term + r_term)

    def _deriv_level(b_lower, d):
        l_d = tt[:, d:-1] - tt[:, : -d - 1]
        r_d = tt[:, d + 1 :] - tt[:, 1:-d]
        return d * (
            _safe_divide(b_lower[:, :-1], l_d) - _safe_divide(b_lower[:, 1:], r_d)
        )

    deriv1 = _deriv_level(b[p - 1], p)
    if p >= 2:
        if p >= 3:
            deriv1_pm2 = _deriv_level(b[p - 3], p - 2)
            deriv2_pm1 = _deriv_level(deriv1_pm2, p - 1)
            deriv3 = _deriv_level(deriv2_pm1, p)
        else:
            deriv3 = jnp.zeros_like(deriv1)
        deriv1_pm1 = _deriv_level(b[p - 2], p - 1)
        deriv2 = _deriv_level(deriv1_pm1, p)
    else:
        deriv2 = jnp.zeros_like(deriv1)

    return b[-1], deriv1, deriv2, deriv3
