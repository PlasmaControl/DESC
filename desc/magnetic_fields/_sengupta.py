"""Exact vacuum fields with nested toroidal surfaces and no continuous symmetry.

From Sengupta et al., "Exact Nonsymmetric Vacuum Magnetic Fields with Nested Toroidal
Surfaces", arXiv:2610.28466 (2026).

"""

import functools

import numpy as np
from scipy.optimize import least_squares
from scipy.special import ellipj, ellipk, roots_legendre

from desc.backend import custom_jvp, fori_loop, jax, jit, jnp, while_loop
from desc.grid import LinearGrid
from desc.utils import errorif, rpz2xyz, xyz2rpz_vec

from ._core import _MagneticField


@functools.lru_cache
def _agm(m):
    """Arithmetic-geometric mean sequences a_n and c_n for the parameter m."""
    a, b, c = [1.0], [np.sqrt(1.0 - m)], [np.sqrt(m)]
    while abs(c[-1]) > np.finfo(float).eps * a[-1] and len(a) < 32:
        a, b, c = (
            a + [(a[-1] + b[-1]) / 2],
            b + [np.sqrt(a[-1] * b[-1])],
            c + [(a[-1] - b[-1]) / 2],
        )
    return tuple(a), tuple(c)


@functools.partial(custom_jvp, nondiff_argnums=(1,))
def _ellipj(u, m):
    """Jacobi elliptic functions sn, cn and dn of real u for a fixed 0 <= m < 1.

    Uses the descending Landen transformation (Abramowitz and Stegun 16.4).
    """
    a, c = _agm(float(m))
    n = len(a) - 1
    phi = 2.0**n * a[-1] * jnp.asarray(u, dtype=float)
    for j in range(n, 0, -1):
        phi = (phi + jnp.arcsin(c[j] / a[j] * jnp.sin(phi))) / 2
    sn = jnp.sin(phi)
    return sn, jnp.cos(phi), jnp.sqrt(1 - m * sn**2)


@_ellipj.defjvp
def _ellipj_jvp(m, primals, tangents):
    (u,), (du,) = primals, tangents
    sn, cn, dn = _ellipj(u, m)
    return (sn, cn, dn), (cn * dn * du, -sn * dn * du, -m * sn * cn * du)


def _sn(x, y, m):
    """Real and imaginary parts of sn(x + iy | m) for real x, y (A&S 16.21.2)."""
    s, c, d = _ellipj(x, m)
    s1, c1, d1 = _ellipj(y, 1 - m)
    den = c1**2 + m * (s * s1) ** 2
    return s * d1 / den, c * d * s1 * c1 / den


def _ellipkinc(phi, m):
    """Incomplete elliptic integral of the first kind F(phi | m) for real phi."""
    n = jnp.round(phi / jnp.pi)
    s, c = jnp.sin(phi - n * jnp.pi), jnp.cos(phi - n * jnp.pi)
    # Carlson's R_F(c^2, 1 - m s^2, 1) by the duplication theorem
    x, y, z = c**2, 1 - m * s**2, jnp.ones_like(c)
    for _ in range(10):
        lam = jnp.sqrt(x * y) + jnp.sqrt(y * z) + jnp.sqrt(z * x)
        x, y, z = (x + lam) / 4, (y + lam) / 4, (z + lam) / 4
    mean = (x + y + z) / 3
    dx, dy = 1 - x / mean, 1 - y / mean
    dz = -dx - dy
    e2, e3 = dx * dy - dz**2, dx * dy * dz
    rf = (1 - e2 / 10 + e3 / 14 + e2**2 / 24 - 3 * e2 * e3 / 44) / jnp.sqrt(mean)
    return 2 * n * ellipk(m) + s * rf


def _bisect(fun, target, hi, n=60):
    """Root of fun(t) = target on [0, hi] for fun increasing, by bisection.

    Points where fun is nan count as above the target.
    """
    lo = jnp.zeros_like(hi)

    def body(i, bounds):
        lo, hi = bounds
        mid = (lo + hi) / 2
        inside = fun(mid) < target
        return jnp.where(inside, mid, lo), jnp.where(inside, hi, mid)

    lo, hi = fori_loop(0, n, body, (lo, hi))
    return (lo + hi) / 2


class SenguptaVacuumField(_MagneticField):
    """Exact vacuum field with nested toroidal surfaces and no continuous symmetry.

    From Sengupta et al., "Exact Nonsymmetric Vacuum Magnetic Fields with Nested
    Toroidal Surfaces", arXiv:2610.28466 (2026). The field is B = ∇Φ with

        Φ = κ⁻¹ Im[F(z₁) + F(z₂)],   F(z) = k⁻¹ log[dn(z|m) − k cn(z|m)],

    z₁ = K + κ(u − u₀) + iτ, z₂ = K + κ(v − v₀) + i(K' + τ), τ = κζ + δ and m = k²,
    where (u, v, ζ) are curvilinear coordinates on a web of confocal ellipsoids or
    paraboloids (the paper's appendix A). Its field lines lie on the level sets of an
    exact label ψ (the paper's eq. 2), which has its minimum ψ₀ = 1/k − 1 on the
    magnetic axis u = u₀, v = v₀. The scale κ is set by the period condition
    κL = 4NK', where L is the period of ζ, so the field is single valued. Every member
    has two field periods.

    The field is only defined in the regular domain of the web around the axis, and is
    nan at points outside it. Lengths are in meters and the field is in Tesla.

    Parameters
    ----------
    k : float
        Modulus of the Jacobi elliptic profiles, 0 < k < 1.
    u0, v0 : float
        Location of the magnetic axis in the (u, v) coordinates of the web.
    m_g : float, optional
        Shape parameter of the confocal ellipsoids, 0 < m_g < 1. Required for the
        ellipsoidal geometry and unused for the paraboloidal geometry.
    delta : float, optional
        Phase of the profiles in ζ. Default 0.
    N : int, optional
        Number of repetitions of the profiles per period of ζ. Default 1.
    geometry : {"ellipsoidal", "paraboloidal"}
        Which web of confocal quadrics to use.

    Examples
    --------
    The paraboloidal member with constant rotational transform 1/√5 has
    K'/K = 5/2, u₀ = arcosh(√5 sech(2π/5)) and v₀ = δ = 0.
    The ellipsoidal member with axis transform √2/7 has k = 3 − 2√2, m_g = k²,
    u₀ = v₀ = 2 F(arcsin(λ₀^(−1/2)) | m_g) with λ₀ = 21√2 − 28, and δ = 0:

    .. code-block:: python

        k = 3 - 2 * np.sqrt(2)
        lambda0 = 21 * np.sqrt(2) - 28
        u0 = 2 * scipy.special.ellipkinc(np.arcsin(lambda0**-0.5), k**2)
        field = SenguptaVacuumField(k=k, u0=u0, v0=u0, m_g=k**2)
        surface = field.get_surface(field.psi0 + 0.1, M=8, N=8)
        Psi = field.compute_toroidal_flux(field.psi0 + 0.1)

    """

    _io_attrs_ = _MagneticField._io_attrs_ + [
        "_k",
        "_u0",
        "_v0",
        "_m_g",
        "_delta",
        "_N",
        "_geometry",
    ]
    _static_attrs = _MagneticField._static_attrs + [
        "_k",
        "_u0",
        "_v0",
        "_m_g",
        "_delta",
        "_N",
        "_geometry",
    ]

    def __init__(self, k, u0, v0, m_g=None, delta=0.0, N=1, geometry="ellipsoidal"):
        errorif(
            geometry not in ["ellipsoidal", "paraboloidal"],
            ValueError,
            f"geometry must be 'ellipsoidal' or 'paraboloidal', got {geometry}",
        )
        errorif(not 0 < k < 1, ValueError, f"k must be in (0, 1), got {k}")
        errorif(
            geometry == "ellipsoidal" and (m_g is None or not 0 < m_g < 1),
            ValueError,
            f"ellipsoidal geometry needs m_g in (0, 1), got {m_g}",
        )
        errorif(int(N) != N or N < 1, ValueError, f"N must be a positive int, got {N}")
        self._k = float(k)
        self._u0 = float(u0)
        self._v0 = float(v0)
        self._m_g = float(m_g) if geometry == "ellipsoidal" else 0.0
        self._delta = float(delta)
        self._N = int(N)
        self._geometry = geometry

    @property
    def NFP(self):
        """int: Number of (toroidal) field periods."""
        return 2

    @property
    def geometry(self):
        """str: Web of confocal quadrics, ellipsoidal or paraboloidal."""
        return self._geometry

    @property
    def psi0(self):
        """float: Value of the surface label ψ on the magnetic axis."""
        return 1 / self._k - 1

    @property
    def kappa(self):
        """float: Scale κ of the profiles, from the period condition κL = 4NK'."""
        return 4 * self._N * ellipk(1 - self._k**2) / self._period

    @property
    def sym(self):
        """bool: Whether the field is stellarator symmetric."""
        Kp = ellipk(1 - self._k**2)
        if self._geometry == "ellipsoidal":
            phase = (self._delta - (self._N - 1) * Kp / 2) / Kp
            return bool(
                np.isclose(self._u0, self._v0) and np.isclose(phase, round(phase))
            )
        return bool(
            np.isclose(self._v0, 0)
            and np.isclose(self._delta / Kp, round(self._delta / Kp))
        )

    @property
    def _period(self):
        """float: Period L of ζ."""
        if self._geometry == "ellipsoidal":
            return 8 * ellipk(1 - self._m_g)
        return 2 * np.pi

    @property
    def _rho_max(self):
        """float: Radius of the largest disc in (u, v) about the axis in the domain."""
        u0, v0 = self._u0, self._v0
        if self._geometry == "ellipsoidal":
            Kg = ellipk(self._m_g)
            web = min(u0 + v0, 4 * Kg - u0 - v0, 4 * Kg - abs(u0 - v0)) / np.sqrt(2)
        else:
            web = (u0 - abs(v0)) / np.sqrt(2)
        return min(ellipk(self._k**2) / self.kappa, web)

    def _position(self, u, v, zeta):
        """Cartesian position X(u, v, ζ), the paper's eqs. A3 and A5."""
        if self._geometry == "ellipsoidal":
            m_t = 1 - self._m_g
            sr, cr, dr = _ellipj((u + v) / 4, self._m_g)
            ss, cs, ds = _ellipj((u - v) / 4, self._m_g)
            st, ct, dt = _ellipj(zeta / 2 + ellipk(m_t) / 2, m_t)
            x = jnp.stack([cr * st * ds, dr * ct * cs, dt * ss], axis=-1)
            return x / sr[..., None]
        return jnp.stack(
            [
                jnp.sinh(u) * jnp.cos(zeta) - jnp.sinh(v) * jnp.sin(zeta),
                jnp.sinh(u) * jnp.sin(zeta) - jnp.sinh(v) * jnp.cos(zeta),
                jnp.sinh(u) * jnp.sinh(v) + jnp.sin(2 * zeta) / 2,
            ],
            axis=-1,
        )

    def _jacobian(self, u, v, zeta):
        """Position and the matrix with columns X_u, X_v, X_ζ."""
        one, zero = jnp.ones_like(u), jnp.zeros_like(u)
        x, xu = jax.jvp(self._position, (u, v, zeta), (one, zero, zero))
        xv = jax.jvp(self._position, (u, v, zeta), (zero, one, zero))[1]
        xz = jax.jvp(self._position, (u, v, zeta), (zero, zero, one))[1]
        return x, jnp.stack([xu, xv, xz], axis=-1)

    def _covariant(self, u, v, zeta):
        """Derivatives Φ_u, Φ_v, Φ_ζ = Im sn(z₁), Im sn(z₂), Re[sn(z₁) + sn(z₂)]."""
        m, kappa = self._k**2, self.kappa
        K, Kp = ellipk(m), ellipk(1 - m)
        tau = kappa * zeta + self._delta
        re1, im1 = _sn(K + kappa * (u - self._u0), tau, m)
        re2, im2 = _sn(K + kappa * (v - self._v0), Kp + tau, m)
        return jnp.stack([im1, im2, re1 + re2], axis=-1)

    def _label(self, u, v, zeta):
        """Surface label ψ, the paper's eq. 2."""
        m, kappa = self._k**2, self.kappa
        _, cx, dx = _ellipj(kappa * (u - self._u0), m)
        _, cy, dy = _ellipj(kappa * (v - self._v0), m)
        U, V = cx / dx, dy / (self._k * cy)
        s, c, d = _ellipj(kappa * zeta + self._delta, 1 - m)
        return (
            d
            * (V - U)
            / jnp.sqrt((c**2 + m * (U * s) ** 2) * (c**2 + m * (V * s) ** 2))
        )

    def _axis_zeta(self, phi):
        """ζ of the point of the magnetic axis with cylindrical azimuth phi."""
        phi = jnp.asarray(phi, dtype=float)
        if self._geometry == "ellipsoidal":
            # the axis is X ∝ (a sn τ, b cn τ, ·) with τ = ζ/2 + K(1 - m_g)/2
            m_t = 1 - self._m_g
            sr, cr, dr, _ = ellipj((self._u0 + self._v0) / 4, self._m_g)
            ss, cs, ds, _ = ellipj((self._u0 - self._v0) / 4, self._m_g)
            a, b = cr * ds / sr, dr * cs / sr
            amplitude = jnp.arctan2(jnp.cos(phi) / a, jnp.sin(phi) / b)
            return 2 * _ellipkinc(amplitude, m_t) - ellipk(m_t)
        # the axis is X = (a cos ζ - b sin ζ, a sin ζ - b cos ζ, ·), so ζ = phi if b = 0
        u0, v0 = jnp.full_like(phi, self._u0), jnp.full_like(phi, self._v0)

        def azimuth(zeta):
            x = self._position(u0, v0, zeta)
            return jnp.arctan2(x[..., 1], x[..., 0])

        zeta = phi
        for _ in range(10):
            f, df = jax.jvp(azimuth, (zeta,), (jnp.ones_like(zeta),))
            zeta = zeta - ((f - phi + np.pi) % (2 * np.pi) - np.pi) / df
        return zeta

    def _coordinates(self, xyz, tol=1e-12, maxiter=50):
        """(u, v, ζ) at Cartesian points, by Newton's method from the axis.

        The iteration starts from the point of the axis with the same cylindrical
        azimuth, and points outside the regular domain are nan.
        """
        target = jax.lax.stop_gradient(xyz)
        zeta = self._axis_zeta(jnp.arctan2(target[:, 1], target[:, 0]))
        q = jnp.stack(
            [jnp.full_like(zeta, self._u0), jnp.full_like(zeta, self._v0), zeta],
            axis=-1,
        )

        def step(q, target):
            x, jac = self._jacobian(*q.T)
            return jnp.linalg.solve(jac, (x - target)[..., None])[..., 0]

        def cond(state):
            i, _, dq = state
            return (i < maxiter) & jnp.any(jnp.abs(dq) > tol)

        def body(state):
            i, q, _ = state
            dq = step(q, target)
            return i + 1, q - dq, dq

        _, q, _ = while_loop(cond, body, (0, q, jnp.ones_like(q)))
        valid = self._in_domain(*q.T) & (
            jnp.linalg.norm(self._position(*q.T) - target, axis=-1) < 1e-8
        )
        # one more step carries the derivatives, d(u, v, ζ)/dX = (X_u, X_v, X_ζ)⁻¹
        return jnp.where(valid[:, None], q - step(q, xyz), jnp.nan)

    def _in_domain(self, u, v, zeta):
        """Whether (u, v) is inside the regular domain of the profiles and the web."""
        K = ellipk(self._k**2) / self.kappa
        valid = (jnp.abs(u - self._u0) < K) & (jnp.abs(v - self._v0) < K)
        if self._geometry == "ellipsoidal":
            Kg = ellipk(self._m_g)
            return valid & (u + v > 0) & (u + v < 4 * Kg) & (jnp.abs(u - v) < 4 * Kg)
        return valid & (u > jnp.abs(v))

    def compute_coordinates(self, coords, basis="rpz"):
        """Compute the curvilinear coordinates (u, v, ζ) of the web at a set of points.

        Parameters
        ----------
        coords : array-like shape(n,3)
            Nodes in [R,phi,Z] or [X,Y,Z] coordinates.
        basis : {"rpz", "xyz"}
            Basis for input coordinates.

        Returns
        -------
        uvz : ndarray, shape(n,3)
            u, v and ζ at the points. ζ is only defined up to multiples of its period.
            Points outside the regular domain of the web are nan.

        """
        assert basis.lower() in ["rpz", "xyz"]
        coords = jnp.atleast_2d(jnp.asarray(coords, dtype=float))
        if basis.lower() == "rpz":
            coords = rpz2xyz(coords)
        return self._coordinates(coords)

    def compute_flux_label(self, coords, basis="rpz"):
        """Compute the exact surface label ψ at a set of points.

        Parameters
        ----------
        coords : array-like shape(n,3)
            Nodes in [R,phi,Z] or [X,Y,Z] coordinates.
        basis : {"rpz", "xyz"}
            Basis for input coordinates.

        Returns
        -------
        psi : ndarray, shape(n,)
            Surface label at the points, which is ``psi0`` on the magnetic axis and
            constant along field lines.

        """
        return self._label(*self.compute_coordinates(coords, basis).T)

    def compute_magnetic_field(
        self,
        coords,
        params=None,
        basis="rpz",
        source_grid=None,
        transforms=None,
        chunk_size=None,
    ):
        """Compute magnetic field at a set of points.

        Parameters
        ----------
        coords : array-like shape(n,3)
            Nodes to evaluate field at in [R,phi,Z] or [X,Y,Z] coordinates.
        params : dict or array-like of dict, optional
            Unused by this MagneticField class.
        basis : {"rpz", "xyz"}
            Basis for input coordinates and returned magnetic field.
        source_grid : Grid, int or None or array-like, optional
            Unused by this MagneticField class.
        transforms : dict of Transform
            Unused by this MagneticField class.
        chunk_size : int or None
            Unused by this MagneticField class.

        Returns
        -------
        field : ndarray, shape(n,3)
            magnetic field at specified points

        """
        assert basis.lower() in ["rpz", "xyz"]
        coords = jnp.atleast_2d(jnp.asarray(coords, dtype=float))
        q = self.compute_coordinates(coords, basis)
        _, jac = self._jacobian(*q.T)
        # B = ∇Φ = Φ_u ∇u + Φ_v ∇v + Φ_ζ ∇ζ, where ∇u, ∇v, ∇ζ are the rows of jac⁻¹
        B = jnp.linalg.solve(
            jnp.swapaxes(jac, -1, -2), self._covariant(*q.T)[..., None]
        )[..., 0]
        if basis.lower() == "rpz":
            B = xyz2rpz_vec(B, phi=coords[:, 1])
        return B

    def compute_magnetic_vector_potential(
        self,
        coords,
        params=None,
        basis="rpz",
        source_grid=None,
        transforms=None,
        chunk_size=None,
    ):
        """Compute magnetic vector potential at a set of points.

        Not implemented for this field.
        """
        raise NotImplementedError(
            "SenguptaVacuumField does not have vector potential calculation "
            "implemented."
        )

    def _contravariant(self, u, v, zeta):
        """Jacobian determinant of X(u, v, ζ) and (B^u, B^v, B^ζ) times it."""
        _, jac = self._jacobian(u, v, zeta)
        metric = jnp.swapaxes(jac, -1, -2) @ jac
        Bsup = jnp.linalg.solve(metric, self._covariant(u, v, zeta)[..., None])[..., 0]
        g = jnp.linalg.det(jac)
        return g, g[..., None] * Bsup

    def _section(self, psi, theta, zeta):
        """Distance from the axis to the surface ψ along a ray in the (u, v) plane."""
        du, dv = jnp.cos(theta), jnp.sin(theta)
        hi = jnp.full(
            jnp.broadcast_shapes(psi.shape, theta.shape), 0.99 * self._rho_max
        )
        return _bisect(
            lambda t: self._label(self._u0 + t * du, self._v0 + t * dv, zeta + 0 * t),
            psi,
            hi,
        )

    def _check_label(self, psi):
        psi = np.atleast_1d(np.asarray(psi, dtype=float))
        errorif(
            np.any(psi <= self.psi0),
            ValueError,
            f"psi must be greater than its value psi0={self.psi0} on the axis.",
        )
        return psi

    def compute_toroidal_flux(self, psi, ntheta=64, nrho=32):
        """Compute the toroidal flux inside the surface ψ.

        The flux is integrated over a section ζ = 0 by Gauss-Legendre quadrature from
        the axis and trapezoidal quadrature around it, with the orientation of
        increasing cylindrical azimuth phi. This is the ``Psi`` of an Equilibrium
        bounded by that surface.

        Parameters
        ----------
        psi : float or array-like
            Values of the surface label, greater than ``psi0``.
        ntheta, nrho : int
            Number of quadrature nodes around and away from the axis.

        Returns
        -------
        Psi : ndarray
            Toroidal flux in Webers inside each surface.

        """
        psi = self._check_label(psi)
        theta = 2 * np.pi * np.arange(ntheta) / ntheta
        t, w = roots_legendre(nrho)
        t, w = (t + 1) / 2, w / 2
        rho = self._section(psi[:, None], theta[None, :], 0.0)
        r = rho[..., None] * t
        u = self._u0 + r * np.cos(theta)[:, None]
        v = self._v0 + r * np.sin(theta)[:, None]
        _, gB = self._contravariant(u.ravel(), v.ravel(), jnp.zeros(u.size))
        # ∫∫ g B^ζ du dv is the flux in the direction of X_u × X_v, whose sense
        # relative to the azimuth is that of ∇ζ
        x, jac = self._jacobian(*(np.array([[self._u0, self._v0, 0.0]]).T))
        phi_hat = jnp.array([-x[0, 1], x[0, 0], 0.0])
        sense = jnp.sign(jnp.dot(jnp.cross(jac[0, :, 0], jac[0, :, 1]), phi_hat))
        integrand = gB[:, 2].reshape(u.shape) * r * rho[..., None] * w
        return sense * 2 * np.pi * jnp.mean(jnp.sum(integrand, axis=-1), axis=-1)

    def compute_iota(self, psi, n=128):
        """Compute the exact rotational transform on the surface ψ.

        ι = (dχ/dψ)/(dΨ/dψ) from the derivatives of the poloidal flux χ through the
        ribbon v = v₀, u > u₀ and of the toroidal flux Ψ through the section ζ = 0
        (the paper's appendix A), both integrated over their edge on the surface.
        The sign is that of DESC, for which (ρ, θ, ζ) is right handed.

        Parameters
        ----------
        psi : float or array-like
            Values of the surface label, greater than ``psi0``.
        n : int
            Number of quadrature nodes around the section and along the ribbon.

        Returns
        -------
        iota : ndarray
            Rotational transform on each surface.

        """
        psi = self._check_label(psi)[:, None]
        theta = 2 * np.pi * np.arange(n) / n
        zeta = self._period * np.arange(n) / n

        # toroidal flux through the section ζ = 0, over the rays at angle theta
        rho = self._section(psi, theta[None, :], 0.0)
        u = (self._u0 + rho * np.cos(theta)).ravel()
        v = (self._v0 + rho * np.sin(theta)).ravel()
        g, gB = self._contravariant(u, v, jnp.zeros_like(u))
        _, dpsi = jax.jvp(
            self._label,
            (u, v, jnp.zeros_like(u)),
            (
                jnp.tile(np.cos(theta), psi.size),
                jnp.tile(np.sin(theta), psi.size),
                0 * u,
            ),
        )
        toroidal = jnp.mean((gB[:, 2] / dpsi).reshape(rho.shape) * rho, axis=-1)

        # poloidal flux through the ribbon v = v0, u > u0 in the sense of increasing v
        rho = self._section(psi, jnp.zeros_like(zeta)[None, :], zeta[None, :])
        u = (self._u0 + rho).ravel()
        zeta = jnp.tile(zeta, psi.size)
        _, dpsi = jax.jvp(
            self._label,
            (u, jnp.full_like(u, self._v0), zeta),
            (jnp.ones_like(u), jnp.zeros_like(u), jnp.zeros_like(u)),
        )
        _, gB = self._contravariant(u, jnp.full_like(u, self._v0), zeta)
        poloidal = jnp.mean((gB[:, 1] / dpsi).reshape(rho.shape), axis=-1)
        poloidal = poloidal * self._period / (2 * np.pi)

        # field lines turn poloidal/toroidal times from u towards v per turn of ζ, which
        # is DESC's θ when (u, v, ζ) is right handed
        return jnp.sign(g[0]) * poloidal / toroidal

    def _axis(self, phi):
        """Cartesian points on the axis at azimuths phi, and ζ there."""
        phi = jnp.asarray(phi, dtype=float)
        zeta = self._axis_zeta(phi)
        return (
            self._position(
                jnp.full_like(phi, self._u0), jnp.full_like(phi, self._v0), zeta
            ),
            zeta,
        )

    def get_axis(self, N=12, nzeta=None):
        """Get the magnetic axis as a FourierRZCurve.

        Parameters
        ----------
        N : int
            Toroidal resolution of the curve.
        nzeta : int, optional
            Number of points in phi over one field period to fit. Default 4N + 4.

        Returns
        -------
        axis : FourierRZCurve
            Magnetic axis.

        """
        from desc.geometry import FourierRZCurve

        nzeta = 4 * N + 4 if nzeta is None else nzeta
        phi = 2 * np.pi * np.arange(nzeta * self.NFP) / (nzeta * self.NFP)
        x, _ = self._axis(jnp.asarray(phi))
        return FourierRZCurve.from_values(
            np.asarray(x), N=N, NFP=self.NFP, sym=self.sym, basis="xyz"
        )

    def get_surface(
        self, psi, M=8, N=8, ntheta=None, nzeta=None, sym=None, refine=True
    ):
        """Get the surface ψ = psi as a FourierRZToroidalSurface.

        The surface is first fit to points on rays from the axis in planes of
        constant phi, along H^(-1/2) (cos θ, sin θ) where H is the Hessian of ψ in R
        and Z on the axis, so that θ is the polar angle of the elliptical surfaces near
        the axis. Its coefficients are then refined to minimize the error in ψ on the
        surface, which leaves the poloidal angle free.

        Parameters
        ----------
        psi : float
            Value of the surface label, greater than ``psi0``.
        M, N : int
            Poloidal and toroidal resolution of the surface.
        ntheta, nzeta : int, optional
            Number of points in θ and in phi over one field period to fit.
            Default 4M + 4 and 4N + 4.
        sym : bool, optional
            Whether to enforce stellarator symmetry. Default is ``self.sym``.
        refine : bool, optional
            Whether to refine the fit by the error in ψ. Default True.

        Returns
        -------
        surface : FourierRZToroidalSurface
            Surface ψ = psi.

        """
        from desc.geometry import FourierRZToroidalSurface

        psi = float(self._check_label(psi)[0])
        ntheta = 4 * M + 4 if ntheta is None else ntheta
        nzeta = 4 * N + 4 if nzeta is None else nzeta
        sym = self.sym if sym is None else sym
        theta, phi = np.meshgrid(
            2 * np.pi * np.arange(ntheta) / ntheta,
            2 * np.pi * np.arange(nzeta) / (nzeta * self.NFP),
            indexing="ij",
        )
        theta, phi = theta.ravel(), jnp.asarray(phi.ravel())
        x, zeta = self._axis(phi)
        R0, Z0 = jnp.hypot(x[:, 0], x[:, 1]), x[:, 2]

        # Hessian of ψ in (R, Z) on the axis, where ∇ψ vanishes
        q = jnp.stack(
            [jnp.full_like(phi, self._u0), jnp.full_like(phi, self._v0), zeta]
        )
        hess = jax.vmap(jax.hessian(lambda q: self._label(*q)))(q.T)
        _, jac = self._jacobian(*q)
        e = jnp.stack(
            [
                jnp.stack([jnp.cos(phi), jnp.sin(phi), 0 * phi], axis=-1),
                jnp.stack([0 * phi, 0 * phi, 1 + 0 * phi], axis=-1),
            ],
            axis=-1,
        )
        grad = jnp.linalg.solve(jac, e)  # derivatives of (u, v, ζ) in R and Z
        hess = jnp.swapaxes(grad, -1, -2) @ hess @ grad
        w, vec = jnp.linalg.eigh(hess)
        ray = vec @ (w[..., None] ** -0.5 * jnp.swapaxes(vec, -1, -2))
        ray = ray @ jnp.stack([np.cos(theta), np.sin(theta)], axis=-1)[..., None]
        dR, dZ = ray[:, 0, 0], ray[:, 1, 0]

        def label(t):
            return self.compute_flux_label(
                jnp.stack([R0 + t * dR, phi, Z0 + t * dZ], axis=-1)
            )

        # ψ - ψ0 ≈ t²/2 near the axis
        hi = jnp.full_like(phi, 2 * np.sqrt(2 * (psi - self.psi0)))
        for _ in range(4):
            hi = jnp.where(label(hi) < psi, 1.5 * hi, hi)
        t = _bisect(label, psi, hi)
        coords = np.stack([R0 + t * dR, phi, Z0 + t * dZ], axis=-1)

        surf = FourierRZToroidalSurface.from_values(
            coords, theta, M=M, N=N, NFP=self.NFP, sym=sym, check_orientation=False
        )
        if surf._compute_orientation() == -1:
            surf = FourierRZToroidalSurface.from_values(
                coords, -theta, M=M, N=N, NFP=self.NFP, sym=sym, check_orientation=False
            )
        if not refine:
            return surf

        # The fit above puts each point at a given θ. Fitting only the shape, by the
        # error in ψ on a finer grid, leaves θ free and is much more accurate.
        grid = LinearGrid(M=2 * M + 2, N=2 * N + 2, NFP=self.NFP)
        AR, AZ = surf.R_basis.evaluate(grid), surf.Z_basis.evaluate(grid)
        nR = surf.R_lmn.size

        def residual(c):
            R, Z = AR @ c[:nR], AZ @ c[nR:]
            x = jnp.stack([R, grid.nodes[:, 2], Z], axis=-1)
            return (self.compute_flux_label(x) - psi) / (psi - self.psi0)

        out = least_squares(
            jit(residual),
            np.concatenate([surf.R_lmn, surf.Z_lmn]),
            jac=jit(jax.jacfwd(residual)),
            method="lm",
            xtol=1e-15,
            ftol=1e-15,
            gtol=1e-15,
            max_nfev=200,
        )
        surf.R_lmn, surf.Z_lmn = out.x[:nR], out.x[nR:]
        return surf
