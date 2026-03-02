"""Functions needed by other tests for computing differences between equilibria."""

import os
import warnings

import numpy as np
from shapely.geometry import Polygon

from desc.backend import put
from desc.derivatives import _Derivative
from desc.grid import Grid, LinearGrid
from desc.utils import ensure_tuple
from desc.vmec import VMECIO


def compute_coords(equil, Nr=10, Nt=8, Nz=None):
    """Computes coordinate values from a given equilibrium."""
    if Nz is None and equil.N == 0:
        Nz = 1
    elif Nz is None:
        Nz = 6

    num_theta = 1000
    num_rho = 1000

    # flux surfaces to plot
    rr = np.linspace(1, 0, Nr, endpoint=False)[::-1]
    rt = np.linspace(0, 2 * np.pi, num_theta)
    rz = np.linspace(0, 2 * np.pi / equil.NFP, Nz, endpoint=False)
    r_grid = LinearGrid(rho=rr, theta=rt, zeta=rz, NFP=equil.NFP)

    # straight field-line angles to plot
    tr = np.linspace(0, 1, num_rho)
    tt = np.linspace(0, 2 * np.pi, Nt, endpoint=False)
    tz = np.linspace(0, 2 * np.pi / equil.NFP, Nz, endpoint=False)
    t_grid = LinearGrid(rho=tr, theta=tt, zeta=tz, NFP=equil.NFP)

    # Note: theta* (also known as vartheta) is the poloidal straight field-line
    # angle in PEST-like flux coordinates

    # find theta angles corresponding to desired theta* angles
    v_grid = Grid(
        equil.map_coordinates(t_grid.nodes, inbasis=("rho", "theta_PEST", "zeta"))
    )
    r_coords = equil.compute(["R", "Z"], grid=r_grid)
    v_coords = equil.compute(["R", "Z"], grid=v_grid)

    # rho contours
    Rr1 = r_coords["R"].reshape(
        (r_grid.num_theta, r_grid.num_rho, r_grid.num_zeta), order="F"
    )
    Rr1 = np.swapaxes(Rr1, 0, 1)
    Zr1 = r_coords["Z"].reshape(
        (r_grid.num_theta, r_grid.num_rho, r_grid.num_zeta), order="F"
    )
    Zr1 = np.swapaxes(Zr1, 0, 1)

    # vartheta contours
    Rv1 = v_coords["R"].reshape(
        (t_grid.num_theta, t_grid.num_rho, t_grid.num_zeta), order="F"
    )
    Rv1 = np.swapaxes(Rv1, 0, 1)
    Zv1 = v_coords["Z"].reshape(
        (t_grid.num_theta, t_grid.num_rho, t_grid.num_zeta), order="F"
    )
    Zv1 = np.swapaxes(Zv1, 0, 1)

    return Rr1, Zr1, Rv1, Zv1


def area_difference(Rr1, Rr2, Zr1, Zr2, Rv1, Rv2, Zv1, Zv2):
    """Compute area difference between coordinate curves.

    Parameters
    ----------
    args : ndarray
        R and Z coordinates of constant rho (r) or vartheta (v) contours.
        Arrays should be indexed as [rho,theta,zeta]

    Returns
    -------
    area_rho : ndarray, shape(Nz, Nr)
        normalized area difference of rho contours, computed as the symmetric
        difference divided by the intersection
    area_theta : ndarray, shape(Nt, Nz)
        normalized area difference between vartheta contours, computed as the area
        of the polygon created by closing the two vartheta contours divided by the
        perimeter squared
    """
    assert Rr1.shape == Rr2.shape == Zr1.shape == Zr2.shape
    assert Rv1.shape == Rv2.shape == Zv1.shape == Zv2.shape

    poly_r1 = np.array(
        [
            [Polygon(np.array([R, Z]).T) for R, Z in zip(Rr1[:, :, i], Zr1[:, :, i])]
            for i in range(Rr1.shape[2])
        ]
    )
    poly_r2 = np.array(
        [
            [Polygon(np.array([R, Z]).T) for R, Z in zip(Rr2[:, :, i], Zr2[:, :, i])]
            for i in range(Rr2.shape[2])
        ]
    )
    poly_v = np.array(
        [
            [
                Polygon(np.array([R, Z]).T)
                for R, Z in zip(
                    np.hstack([Rv1[:, :, i].T, Rv2[::-1, :, i].T]),
                    np.hstack([Zv1[:, :, i].T, Zv2[::-1, :, i].T]),
                )
            ]
            for i in range(Rv1.shape[2])
        ]
    )

    diff_rho = np.array(
        [
            poly1.symmetric_difference(poly2).area
            for poly1, poly2 in zip(poly_r1.flat, poly_r2.flat)
        ]
    ).reshape((Rr1.shape[2], Rr1.shape[0]))
    # for some reason shapely sometimes throws a warning here on CI but not locally,
    # see https://github.com/shapely/shapely/issues/1345
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="invalid value encountered in intersection"
        )
        intersect_rho = np.array(
            [
                poly1.intersection(poly2).area
                for poly1, poly2 in zip(poly_r1.flat, poly_r2.flat)
            ]
        ).reshape((Rr1.shape[2], Rr1.shape[0]))
    area_rho = np.where(
        diff_rho > 0, diff_rho / np.where(intersect_rho != 0, intersect_rho, 1), 0
    )
    area_theta = np.array(
        [poly.area / (poly.length) ** 2 for poly in poly_v.flat]
    ).reshape((Rv1.shape[1], Rv1.shape[2]))
    return area_rho, area_theta


def area_difference_vmec(equil, vmec_data, Nr=10, Nt=8, Nz=None, **kwargs):
    """Compute average normalized area difference between VMEC and DESC equilibria.

    Parameters
    ----------
    equil : Equilibrium
        desc equilibrium to compare
    vmec_data : dict
        dictionary of vmec outputs
    Nr : int, optional
        number of radial surfaces to average over
    Nt : int, optional
        number of vartheta contours to compare
    Nz : int, optional
        Number of zeta planes to compare. If None, use 1 plane for axisymmetric cases
        or 6 for non-axisymmetric.

    Returns
    -------
    area_rho : ndarray, shape(Nz, Nr)
        normalized area difference of rho contours, computed as the symmetric
        difference divided by the intersection
    area_theta : ndarray, shape(Nt, Nz)
        normalized area difference between vartheta contours, computed as the area
        of the polygon created by closing the two vartheta contours divided by the
        perimeter squared

    """
    # 1e-3 tolerance seems reasonable for testing, similar to comparison by eye
    if isinstance(vmec_data, (str, os.PathLike)):
        vmec_data = VMECIO.read_vmec_output(vmec_data)

    signgs = vmec_data["signgs"]
    coords = VMECIO.compute_coord_surfaces(equil, vmec_data, Nr, Nt, Nz, **kwargs)

    Rr1 = coords["Rr_desc"]
    Zr1 = coords["Zr_desc"]
    Rv1 = coords["Rv_desc"]
    Zv1 = coords["Zv_desc"]
    Rr2 = coords["Rr_vmec"]
    Zr2 = coords["Zr_vmec"]
    # need to reverse the order of these due to different sign conventions for theta
    Rv2 = coords["Rv_vmec"][::signgs]
    Zv2 = coords["Zv_vmec"][::signgs]
    area_rho, area_theta = area_difference(Rr1, Rr2, Zr1, Zr2, Rv1, Rv2, Zv1, Zv2)
    return area_rho, area_theta


def area_difference_desc(eq1, eq2, Nr=10, Nt=8, Nz=None):
    """Compute average normalized area difference between two DESC equilibria.

    Parameters
    ----------
    eq1, eq2 : Equilibrium
        desc equilibria to compare
    Nr : int, optional
        Number of radial surfaces to average over
    Nt : int, optional
        Number of vartheta contours to compare
    Nz : int, optional
        Number of zeta planes to compare. If None, use 1 plane for axisymmetric cases
        or 6 for non-axisymmetric.

    Returns
    -------
    area_rho : ndarray, shape(Nr, Nz)
        normalized area difference of rho contours, computed as the symmetric
        difference divided by the intersection
    area_theta : ndarray, shape(Nt, Nz)
        normalized area difference between vartheta contours, computed as the area
        of the polygon created by closing the two vartheta contours divided by the
        perimeter squared

    """
    Rr1, Zr1, Rv1, Zv1 = compute_coords(eq1, Nr=Nr, Nt=Nt, Nz=Nz)
    Rr2, Zr2, Rv2, Zv2 = compute_coords(eq2, Nr=Nr, Nt=Nt, Nz=Nz)

    area_rho, area_theta = area_difference(Rr1, Rr2, Zr1, Zr2, Rv1, Rv2, Zv1, Zv2)
    return area_rho, area_theta


class FiniteDiffDerivative(_Derivative):
    """Computes derivatives using 2nd order centered finite differences.

    Parameters
    ----------
    fun : callable
        Function to be differentiated.
    argnum : int, optional
        Specifies which positional argument to differentiate with respect to
    mode : str, optional
        Automatic differentiation mode.
        One of ``'fwd'`` (forward mode Jacobian), ``'rev'`` (reverse mode Jacobian),
        ``'grad'`` (gradient of a scalar function),
        ``'hess'`` (Hessian of a scalar function),
        or ``'jvp'`` (Jacobian vector product)
        Default = ``'fwd'``
    rel_step : float, optional
        Relative step size: dx = max(1, abs(x))*rel_step
        Default = 1e-3

    """

    def __init__(self, fun, argnum=0, mode="fwd", rel_step=1e-3, **kwargs):

        self._fun = fun
        self._argnum = argnum
        self.rel_step = rel_step
        self._set_mode(mode)

    def _compute_hessian(self, *args, **kwargs):
        """Compute the Hessian matrix using 2nd order centered finite differences.

        Parameters
        ----------
        args : tuple
            Arguments of the objective function where the derivative is to be
            evaluated at.
        kwargs : dict
            keyword arguments passed to fun

        Returns
        -------
        H : ndarray of float, shape(len(x),len(x))
            d^2f/dx^2, where f is the output of the function fun and x is the input
            argument at position argnum.

        """

        def f(x):
            tempargs = args[0 : self._argnum] + (x,) + args[self._argnum + 1 :]
            return self._fun(*tempargs, **kwargs)

        x = np.atleast_1d(args[self._argnum])
        n = len(x)
        fx = f(x)
        h = np.maximum(1.0, np.abs(x)) * self.rel_step
        ee = np.diag(h)
        hess = np.outer(h, h)

        for i in range(n):
            eei = ee[i, :]
            hess[i, i] = (f(x + 2 * eei) - 2 * fx + f(x - 2 * eei)) / (4.0 * hess[i, i])
            for j in range(i + 1, n):
                eej = ee[j, :]
                hess[i, j] = (
                    f(x + eei + eej)
                    - f(x + eei - eej)
                    - f(x - eei + eej)
                    + f(x - eei - eej)
                ) / (4.0 * hess[j, i])
                hess[j, i] = hess[i, j]

        return hess

    def _compute_grad_or_jac(self, *args, **kwargs):
        """Compute the gradient or Jacobian matrix (ie, first derivative).

        Parameters
        ----------
        args : tuple
            Arguments of the objective function where the derivative is to be
            evaluated at.
        kwargs : dict
            keyword arguments passed to fun

        Returns
        -------
        J : ndarray of float, shape(len(f),len(x))
            df/dx, where f is the output of the function fun and x is the input
            argument at position argnum.

        """

        def f(x):
            tempargs = args[0 : self._argnum] + (x,) + args[self._argnum + 1 :]
            return self._fun(*tempargs, **kwargs)

        x0 = np.atleast_1d(args[self._argnum])
        f0 = f(x0)
        m = f0.size
        n = x0.size
        J = np.zeros((m, n))
        h = np.maximum(1.0, np.abs(x0)) * self.rel_step
        h_vecs = np.diag(np.atleast_1d(h))
        for i in range(n):
            x1 = x0 - h_vecs[i]
            x2 = x0 + h_vecs[i]
            dx = x2[i] - x1[i]
            f1 = f(x1)
            f2 = f(x2)
            df = f2 - f1
            dfdx = df / dx
            J = put(J.T, i, dfdx.flatten()).T
        if m == 1:
            J = np.ravel(J)
        return J

    @classmethod
    def compute_vjp(cls, fun, argnum, v, *args, **kwargs):
        """Compute v.T * df/dx.

        Parameters
        ----------
        fun : callable
            function to differentiate
        argnum : int or tuple
            arguments to differentiate with respect to
        v : array-like or tuple of array-like
            tangent vectors. Should be one for each output of fun
        args : tuple
            arguments passed to fun
        kwargs : dict
            keyword arguments passed to fun

        Returns
        -------
        vjp : array-like
            Vector v times Jacobian, summed over different argnums

        """
        assert np.isscalar(argnum), "vjp for multiple args not currently supported"
        rel_step = kwargs.pop("rel_step", 1e-3)

        def _fun(*args):
            return v.T @ fun(*args, **kwargs)

        return FiniteDiffDerivative(_fun, argnum, "grad", rel_step)(*args)

    @classmethod
    def compute_jvp(cls, fun, argnum, v, *args, **kwargs):
        """Compute df/dx*v.

        Parameters
        ----------
        fun : callable
            function to differentiate
        argnum : int or tuple
            arguments to differentiate with respect to
        v : array-like or tuple of array-like
            tangent vectors. Should be one for each argnum
        args : tuple
            arguments passed to fun
        kwargs : dict
            keyword arguments passed to fun

        Returns
        -------
        jvp : array-like
            Jacobian times vectors v, summed over different argnums

        """
        rel_step = kwargs.pop("rel_step", 1e-3)

        if np.isscalar(argnum):
            nargs = 1
            argnum = (argnum,)
        else:
            nargs = len(argnum)
        v = ensure_tuple(v)

        f = np.array(
            [
                cls._compute_jvp_1arg(
                    fun, argnum[i], v[i], *args, rel_step=rel_step, **kwargs
                )
                for i in range(nargs)
            ]
        )
        return np.sum(f, axis=0)

    @classmethod
    def compute_jvp2(cls, fun, argnum1, argnum2, v1, v2, *args, **kwargs):
        """Compute d^2f/dx^2*v1*v2.

        Parameters
        ----------
        fun : callable
            function to differentiate
        argnum1, argnum2 : int or tuple of int
            arguments to differentiate with respect to. First entry corresponds to v1,
            second to v2
        v1,v2 : array-like or tuple of array-like
            tangent vectors. Should be one for each argnum
        args : tuple
            arguments passed to fun
        kwargs : dict
            keyword arguments passed to fun

        Returns
        -------
        jvp2 : array-like
            second derivative times vectors v1, v2, summed over different argnums

        """
        if np.isscalar(argnum1):
            v1 = ensure_tuple(v1)
            argnum1 = (argnum1,)
        else:
            v1 = tuple(v1)

        if np.isscalar(argnum2):
            argnum2 = (argnum2 + 1,)
            v2 = ensure_tuple(v2)
        else:
            argnum2 = tuple([i + 1 for i in argnum2])
            v2 = tuple(v2)

        dfdx = lambda dx1, *args: cls.compute_jvp(fun, argnum1, dx1, *args, **kwargs)
        d2fdx2 = lambda dx1, dx2: cls.compute_jvp(
            dfdx, argnum2, dx2, dx1, *args, **kwargs
        )
        return d2fdx2(v1, v2)

    @classmethod
    def compute_jvp3(cls, fun, argnum1, argnum2, argnum3, v1, v2, v3, *args, **kwargs):
        """Compute d^3f/dx^3*v1*v2*v3.

        Parameters
        ----------
        fun : callable
            function to differentiate
        argnum1, argnum2, argnum3 : int or tuple of int
            arguments to differentiate with respect to. First entry corresponds to v1,
            second to v2 etc
        v1,v2,v3 : array-like or tuple of array-like
            tangent vectors. Should be one for each argnum
        args : tuple
            arguments passed to fun
        kwargs : dict
            keyword arguments passed to fun

        Returns
        -------
        jvp3 : array-like
            third derivative times vectors v2, v3, v3, summed over different argnums

        """
        if np.isscalar(argnum1):
            v1 = ensure_tuple(v1)
            argnum1 = (argnum1,)
        else:
            v1 = tuple(v1)

        if np.isscalar(argnum2):
            argnum2 = (argnum2 + 1,)
            v2 = ensure_tuple(v2)
        else:
            argnum2 = tuple([i + 1 for i in argnum2])
            v2 = tuple(v2)

        if np.isscalar(argnum3):
            argnum3 = (argnum3 + 2,)
            v3 = ensure_tuple(v3)
        else:
            argnum3 = tuple([i + 2 for i in argnum3])
            v3 = tuple(v3)

        dfdx = lambda dx1, *args: cls.compute_jvp(fun, argnum1, dx1, *args, **kwargs)
        d2fdx2 = lambda dx1, dx2, *args: cls.compute_jvp(
            dfdx, argnum2, dx2, dx1, *args, **kwargs
        )
        d3fdx3 = lambda dx1, dx2, dx3: cls.compute_jvp(
            d2fdx2, argnum3, dx3, dx2, dx1, *args, **kwargs
        )
        return d3fdx3(v1, v2, v3)

    def _compute_jvp(self, v, *args, **kwargs):
        return self.compute_jvp(
            self._fun, self._argnum, v, *args, rel_step=self.rel_step, **kwargs
        )

    @classmethod
    def _compute_jvp_1arg(cls, fun, argnum, v, *args, **kwargs):
        """Compute a jvp wrt a single argument."""
        rel_step = kwargs.pop("rel_step", 1e-3)
        normv = np.linalg.norm(v)
        if normv != 0:
            vh = v / normv
        else:
            vh = v
        x = args[argnum]

        def f(x):
            tempargs = args[0:argnum] + (x,) + args[argnum + 1 :]
            return fun(*tempargs, **kwargs)

        h = rel_step
        df = (f(x + h * vh) - f(x - h * vh)) / (2 * h)
        return df * normv

    def _set_mode(self, mode):
        if mode not in ["fwd", "rev", "grad", "hess", "jvp"]:
            raise ValueError(
                "invalid mode option for finite difference differentiation"
            )

        self._mode = mode
        if self._mode == "fwd":
            self._compute = self._compute_grad_or_jac
        elif self._mode == "rev":
            self._compute = self._compute_grad_or_jac
        elif self._mode == "grad":
            self._compute = self._compute_grad_or_jac
        elif self._mode == "hess":
            self._compute = self._compute_hessian
        elif self._mode == "jvp":
            self._compute = self._compute_jvp

    def compute(self, *args, **kwargs):
        """Compute the derivative matrix.

        Parameters
        ----------
        *args : list
            Arguments of the objective function where the derivative is to be
            evaluated at.

        Returns
        -------
        D : ndarray of float
            derivative of f evaluated at x, where f is the output of the function
            fun and x is the input argument at position argnum. Exact shape and meaning
            will depend on "mode"

        """
        return self._compute(*args, **kwargs)
