"""Dommaschk potential utility functions.

based off Representations for vacuum potentials in stellarators
https://doi.org/10.1016/0010-4655(86)90109-8

"""

import sympy as sp
from sympy.abc import x as x_sym
from sympy.vector import CoordSys3D

from desc.backend import jit, jnp
from desc.derivatives import Derivative
from desc.utils import Timer

from ._core import ScalarPotentialField, _MagneticField


class DommaschkPotentialField(ScalarPotentialField):
    """Magnetic field due to a Dommaschk scalar magnetic potential in rpz coordinates.

        From Dommaschk 1986 paper https://doi.org/10.1016/0010-4655(86)90109-8

        this is the field due to the dommaschk potential (eq. 1) for
        a given set of m,l indices and their corresponding
        coefficients a_ml, b_ml, c_ml d_ml.

    Parameters
    ----------
    ms : 1D array-like of int
        first indices of V_m_l terms (eq. 12 of reference),
        corresponds to the toroidal periodicity of the mode.
    ls : 1D array-like of int
        second indices of V_m_l terms (eq. 12 of reference),
        corresponds to the poloidal periodicity of the mode.
    a_arr : 1D array-like of float
        a_m_l coefficients of V_m_l terms, which multiply the cos(m*phi)*D_m_l terms
    b_arr : 1D array-like of float
        b_m_l coefficients of V_m_l terms, which multiply the sin(m*phi)*D_m_l terms
    c_arr : 1D array-like of float
        c_m_l coefficients of V_m_l terms, which multiply the cos(m*phi)*N_m_l-1 term
    d_arr : 1D array-like of float
        d_m_l coefficients of V_m_l terms, which multiply the sin(m*phi)*N_m_l-1 terms
    B0: float
        scale strength of the magnetic field's 1/R portion
    NFP : int, optional
        Whether the field has a discrete periodicity. This is only used when making
        a ``SplineMagneticField`` from this field using its ``from_field`` method,
        or when saving this field as an mgrid file using the ``save_mgrid`` method.

    """

    def __init__(
        self,
        ms=jnp.array([0]),
        ls=jnp.array([0]),
        a_arr=jnp.array([0.0]),
        b_arr=jnp.array([0.0]),
        c_arr=jnp.array([0.0]),
        d_arr=jnp.array([0.0]),
        B0=1.0,
        NFP=1,
    ):
        ms = jnp.atleast_1d(jnp.asarray(ms))
        ls = jnp.atleast_1d(jnp.asarray(ls))
        a_arr = jnp.atleast_1d(jnp.asarray(a_arr))
        b_arr = jnp.atleast_1d(jnp.asarray(b_arr))
        c_arr = jnp.atleast_1d(jnp.asarray(c_arr))
        d_arr = jnp.atleast_1d(jnp.asarray(d_arr))

        assert (
            ms.size == ls.size == a_arr.size == b_arr.size == c_arr.size == d_arr.size
        ), "Passed in arrays must all be of the same size!"
        assert not jnp.any(
            jnp.logical_or(ms < 0, ls < 0)
        ), "m and l mode numbers must be >= 0!"
        assert (
            jnp.isscalar(B0) or jnp.atleast_1d(B0).size == 1
        ), "B0 should be a scalar value!"

        ms_over_NFP = ms / NFP
        assert jnp.allclose(
            ms_over_NFP, ms_over_NFP.astype(int)
        ), "To enforce desired NFP, `ms` should be all integer multiples of NFP"

        params = {}
        params["ms"] = ms
        params["ls"] = ls
        params["a_arr"] = a_arr
        params["b_arr"] = b_arr
        params["c_arr"] = c_arr
        params["d_arr"] = d_arr
        params["B0"] = B0
        self._full_params = params.copy()

        super().__init__(dommaschk_potential, params, NFP)

        self._set_potentials()

    def _set_potentials(self):
        """Creates the potential using sympy."""
        cyl = CoordSys3D(
            "cyl", transformation="cylindrical", variable_names=("R", "phi", "Z")
        )
        keys = list(self._full_params.keys())

        params = dict(
            zip(
                keys,
                [self._full_params[key].tolist() for key in keys if key != "B0"],
            )
        )
        params["a_arr"] = sp.symbols(
            " ".join([f"a_{l}{m}" for l, m in zip(params["ls"], params["ms"])])
        )
        params["b_arr"] = sp.symbols(
            " ".join([f"b_{l}{m}" for l, m in zip(params["ls"], params["ms"])])
        )
        params["c_arr"] = sp.symbols(
            " ".join([f"c_{l}{m}" for l, m in zip(params["ls"], params["ms"])])
        )
        params["d_arr"] = sp.symbols(
            " ".join([f"d_{l}{m}" for l, m in zip(params["ls"], params["ms"])])
        )

        params["B0"] = sp.symbols("B0")

        domm_pot = dommaschk_potential(cyl.R, cyl.phi, cyl.Z, **params)

        potential_fxn = sp.lambdify(
            [
                cyl.R,
                cyl.phi,
                cyl.Z,
                params["a_arr"],
                params["b_arr"],
                params["c_arr"],
                params["d_arr"],
                params["B0"],
            ],
            domm_pot,
            "jax",
        )
        # resulting function from lambdify does not take kwargs, so we need to
        # wrap it in a lambda function that does
        self._potential = lambda r, p, z, a_arr, b_arr, c_arr, d_arr, B0: potential_fxn(
            r,
            p,
            z,
            a_arr,
            b_arr,
            c_arr,
            d_arr,
            B0,
        )

        # remove ms and ls as are no longer needed in params, they are instead baked
        # into the potential function
        self._params.pop("ms")
        self._params.pop("ls")

    @classmethod
    def fit_magnetic_field(  # noqa: C901
        cls,
        field,
        coords,
        max_m,
        max_l,
        sym=False,
        verbose=1,
        NFP=1,
        chunk_size=None,
    ):
        """Fit a vacuum magnetic field with a Dommaschk Potential field.

        Parameters
        ----------
        field (MagneticField or callable or ndarray): magnetic field to fit
            if callable, must accept (num_nodes,3) array of rpz coords as argument
                and output (num_nodes,3) as the B field in cylindrical rpz basis.
            if ndarray, must be an ndarray of the magnetic field in rpz,
                of shape (num_nodes,3) with the columns being (B_R, B_phi, B_Z)
        coords (ndarray): shape (num_nodes,3) of R,phi,Z points to fit field at
        max_m (int): maximum m to use for Dommaschk Potentials, within one field period
            i.e. if NFP= 2 and max_m = 3, then modes with arguments up to 3*2*phi will
            be included
        max_l (int): maximum l to use for Dommaschk Potentials
        sym (bool): if field is stellarator symmetric or not.
            if True, only stellarator-symmetric modes will
            be included in the fitting
        NFP (int): if the field being fit has a discrete toroidal symmetry
            with field period NFP. This will only allow Dommaschk m modes
            that are integer multiples of NFP.
        verbose (int): verbosity level of fitting routine, > 0 prints residuals,
             >1 prints timing info
        chunk_size : int or None
            Size to split computation into chunks of evaluation points.
            If no chunking should be done or the chunk size is the full input
            then supply ``None``. Default is ``None``.

        """
        # We seek c in  Ac = b
        # A will be the BR, Bphi and BZ from each individual
        # dommaschk potential basis function evaluated at each node
        # c is the dommaschk potential coefficients
        # c will be [B0, a_00, a_10, a_01, a_11... etc]
        # b is the magnetic field at each node which we are fitting
        if isinstance(field, _MagneticField):
            B = field.compute_magnetic_field(coords, chunk_size=chunk_size)
        elif callable(field):
            B = field(coords)
        else:  # it must be the field evaluated at the passed-in coords
            B = field
        # TODO (#928): add basis argument for if passed-in field or callable
        # evaluates rpz or xyz basis magnetic field vector,
        # and what basis coords is

        #########
        # make b
        #########
        # we will have the rhs be 3*num_nodes in length (bc of vector B)

        rhs = jnp.vstack((B[:, 0], B[:, 1], B[:, 2])).T.flatten(order="F")

        #####################
        # b is made, now do A
        #####################
        num_modes = 1 + (max_l) * (max_m + 1) * 4
        # TODO (#928): if symmetric, technically only need half the modes
        # however, the field and functions are setup to accept equal
        # length arrays for a,b,c,d, so we will just zero out the
        # modes that don't fit symmetry, but in future
        # should refactor code to have a 3rd index so that
        # we have a = V_ml0, b = V_ml1, c = V_ml2, d = V_ml3
        # and the modes array can then be [m,l,x] where x is 0,1,2,3
        # and we dont need to keep track of a,b,c,d separately

        # TODO (#928): technically we can drop some modes
        # since if max_l=0, there are only ever nonzero terms for a and b
        # and if max_m=0, there are only ever nonzero terms for a and c
        # but since we are only fitting in a least squares sense,
        # and max_l and max_m should probably be both nonzero anyways,
        # this is not an issue right now

        # mode numbers
        ms = []
        ls = []

        # order of coeffs in the vector c are B0, a_ml, b_ml, c_ml, d_ml
        a_s = []
        b_s = []
        c_s = []
        d_s = []
        zero_due_to_sym_inds = []
        abcd_zero_due_to_sym_inds = [
            [],
            [],
            [],
            [],
        ]  # indices that should be 0 due to symmetry
        # if sym is True, when l is even then we need a=d=0
        # and if l is odd then b=c=0

        for l in range(1, max_l + 1):
            for m in range(0, max_m * NFP + 1, NFP):
                if not sym:
                    pass  # no sym, use all coefs
                elif l % 2 == 0:
                    zero_due_to_sym_inds = [0, 3]  # a=d=0 for even l with sym
                elif l % 2 == 1:
                    zero_due_to_sym_inds = [1, 2]  # b=c=0 for odd l with sym
                for which_coef in range(4):
                    if which_coef == 0:
                        a_s.append(1)
                    elif which_coef == 1:
                        b_s.append(1)
                    elif which_coef == 2:
                        c_s.append(1)
                    elif which_coef == 3:
                        d_s.append(1)
                    if which_coef in zero_due_to_sym_inds:
                        abcd_zero_due_to_sym_inds[which_coef].append(0)
                    else:
                        abcd_zero_due_to_sym_inds[which_coef].append(1)
                ms.append(m)
                ls.append(l)
        for i in range(4):
            abcd_zero_due_to_sym_inds[i] = jnp.asarray(abcd_zero_due_to_sym_inds[i])
        assert (len(a_s) + len(b_s) + len(c_s) + len(d_s)) == num_modes - 1

        params = {
            "ms": ms,
            "ls": ls,
            "a_arr": a_s,
            "b_arr": b_s,
            "c_arr": c_s,
            "d_arr": d_s,
            "B0": 0.0,
        }
        n = (
            round(num_modes - 1) / 4
        )  # how many l-m mode pairs there are, also is len(a_s)
        n = int(n)
        timer = Timer()
        timer.start("Construct DommaschkPotentialField Object")
        domm_field = DommaschkPotentialField(**params)
        timer.stop("Construct DommaschkPotentialField Object")
        if verbose > 1:
            timer.disp("Construct DommaschkPotentialField Object")

        def get_B_dom(coords, X):
            """Fxn wrapper to find jacobian of dommaschk B wrt coefs a,b,c,d."""
            # zero out any terms that should be zero due to symmetry, which
            # we cataloged earlier for each a_arr,b_arr,c_arr,d_arr
            # that way the resulting modes after pinv don't contain them either
            return domm_field.compute_magnetic_field(
                coords,
                params={
                    "a_arr": jnp.asarray(X[1 : n + 1]) * abcd_zero_due_to_sym_inds[0],
                    "b_arr": jnp.asarray(X[n + 1 : 2 * n + 1])
                    * abcd_zero_due_to_sym_inds[1],
                    "c_arr": jnp.asarray(X[2 * n + 1 : 3 * n + 1])
                    * abcd_zero_due_to_sym_inds[2],
                    "d_arr": jnp.asarray(X[3 * n + 1 : 4 * n + 1])
                    * abcd_zero_due_to_sym_inds[3],
                    "B0": X[0],
                },
                chunk_size=chunk_size,
            )

        X = []
        for key in ["B0", "a_arr", "b_arr", "c_arr", "d_arr"]:
            obj = params[key]
            if isinstance(obj, list):
                X += obj
            else:
                X += [obj]
        X = jnp.asarray(X)
        timer.start("Compute Jacobian")
        jac = jit(Derivative(get_B_dom, argnum=1))(coords, X)
        timer.stop("Compute Jacobian")
        if verbose > 1:
            timer.disp("Compute Jacobian")

        A = jac.reshape((rhs.size, len(X)), order="F")

        # now solve Ac=b for the coefficients c

        # TODO (#928): use min singular value to give sense of cond number?
        c, res, rank, s = jnp.linalg.lstsq(A, rhs)

        if verbose > 0:
            # res is a list of len(1) so index into it
            print(f"Sum of Squares Residual of fit: {res[0]:1.4e} T^2")
            res_at_pts = A @ c - rhs
            print(f"Max Absolute Residual: {jnp.max(abs(res_at_pts)):1.4e} T")
            print(f"Min Absolute Residual: {jnp.min(abs(res_at_pts)):1.4e} T")
            print(f"Mean Absolute Residual: {jnp.mean(abs(res_at_pts)):1.4e} T")

        # recover the params from the c coefficient vector
        B0 = c[0]

        # we zero out the terms that should be zero due to symmetry here
        # TODO (#928): should also just not return any zeroed-out modes, but
        # the way the modes are cataloged here with the ls and ms arrays,
        # it is not straightforward to do that
        a_arr = c[1 : n + 1] * abcd_zero_due_to_sym_inds[0]
        b_arr = c[n + 1 : 2 * n + 1] * abcd_zero_due_to_sym_inds[1]
        c_arr = c[2 * n + 1 : 3 * n + 1] * abcd_zero_due_to_sym_inds[2]
        d_arr = c[3 * n + 1 : 4 * n + 1] * abcd_zero_due_to_sym_inds[3]

        domm_field._params["B0"] = B0
        domm_field._params["a_arr"] = a_arr
        domm_field._params["b_arr"] = b_arr
        domm_field._params["c_arr"] = c_arr
        domm_field._params["d_arr"] = d_arr
        return domm_field


# Dommaschk potential utility functions
# these kwargs found to make sp.integrate faster
sp_integrate_kwargs = {
    "meijerg": None,
    "heurisch": False,
    "manual": False,
}


def gamma(n):
    """Gamma function, only implemented for integers (equiv to factorial of (n-1))."""
    return sp.factorial(n - 1)


def CD_0_0(R):
    """Eq 8, CD_m_k at m=0 k=0."""
    return 1


def CD_m_0(R, m):
    """Eq 8, CD_m_k at m>0, k=0."""
    return (R**m + R ** (-m)) / 2


def CD_m_k(R, m, k):
    """Eq 6 and 7 of Dommaschk paper, for system in Eq 8."""
    if m == 0:
        if k == 0:
            return CD_0_0(R)
        # call itself recursively
        # Eq 6
        return sp.integrate(
            CD_m_k(x_sym, 0, k - 1) * (sp.log(x_sym) - sp.log(R)) * x_sym,
            (x_sym, 1, R),
            **sp_integrate_kwargs,
        )
    elif k == 0:
        return CD_m_0(R, m)
    else:
        # Eq 7
        return (
            sp.integrate(
                CD_m_k(x_sym, m, k - 1) * ((x_sym / R) ** m - (R / x_sym) ** m) * x_sym,
                (x_sym, 1, R),
                **sp_integrate_kwargs,
            )
            / 2
            / m
        )


def CN_0_0(R):
    """Eq 9, CN_m_k at m=0 k=0."""
    return sp.log(R)


def CN_m_0(R, m):
    """Eq 9, CN_m_k at m>0 k=0."""
    return (R**m - R ** (-m)) / 2 / m


def CN_m_k(R, m, k):
    """Eq 6/7 of Dommaschk paper, for system in Eq 9."""
    if m == 0:
        if k == 0:
            return CN_0_0(R)
        # call itself recursively
        # Eq 6
        return sp.integrate(
            CN_m_k(x_sym, 0, k - 1) * (sp.log(x_sym) - sp.log(R)) * x_sym,
            (x_sym, 1, R),
            **sp_integrate_kwargs,
        )
    elif k == 0:
        return CN_m_0(R, m)
    else:
        # Eq 7
        return (
            sp.integrate(
                CN_m_k(x_sym, m, k - 1) * ((x_sym / R) ** m - (R / x_sym) ** m) * x_sym,
                (x_sym, 1, R),
                **sp_integrate_kwargs,
            )
            / 2
            / m
        )


def D_m_n(R, Z, m, n):
    """D_m_n term in eqn 3 and 8 of Dommaschk paper."""
    result = 0.0
    for k in range(n // 2 + 1):
        # sp.expand here found to make later AD of potential faster
        coef = sp.expand(CD_m_k(R, m, k)) / gamma(n - 2 * k + 1)
        exp = n - 2 * k
        result += coef * Z**exp
    return result


def N_m_n(R, Z, m, n):
    """N_m_n term in eqn 3 and 9 of Dommaschk paper."""
    result = 0.0
    for k in range(n // 2 + 1):
        # sp.expand here found to make later AD of potential faster
        coef = sp.expand(CN_m_k(R, m, k)) / gamma(n - 2 * k + 1)
        exp = n - 2 * k
        result += coef * Z**exp
    return result


def V_m_l(R, phi, Z, m, l, a, b, c, d):
    """Eq 12 of Dommaschk paper.

    Parameters
    ----------
    R,phi,Z : array-like
        Cylindrical coordinates (1-D arrays of each of size num_eval_pts)
            to evaluate the Dommaschk potential term at.
    m : int
        first index of V_m_l term
    l : int
        second index of V_m_l term
    a : float
        a_m_l coefficient of V_m_l term, which multiplies cos(m*phi)*D_m_l
    b : float
        b_m_l coefficient of V_m_l term, which multiplies sin(m*phi)*D_m_l
    c : float
        c_m_l coefficient of V_m_l term, which multiplies cos(m*phi)*N_m_l-1
    d : float
        d_m_l coefficient of V_m_l term, which multiplies sin(m*phi)*N_m_l-1

    Returns
    -------
    value : array-like
        Value of this V_m_l term evaluated at the given R,phi,Z points
        (same size as the size of the given R,phi, or Z arrays).

    """
    return (a * sp.cos(m * phi) + b * sp.sin(m * phi)) * D_m_n(R, Z, m, l) + (
        c * sp.cos(m * phi) + d * sp.sin(m * phi)
    ) * N_m_n(R, Z, m, l - 1)


def dommaschk_potential(R, phi, Z, ms, ls, a_arr, b_arr, c_arr, d_arr, B0=1):
    """Eq 1 of Dommaschk paper.

        this is the total dommaschk potential for
        a given set of m,l indices and their corresponding
        coefficients a_ml, b_ml, c_ml d_ml.

    Parameters
    ----------
    R,phi,Z : sympy.vector.scalar.BaseScalar
        Cylindrical coordinate sympy symbols.
        Obtained from sympy.vector.CoordSys3D using the ``"cyl"`` basis.
    ms : 1D array-like of int
        first indices of V_m_l terms
    ls : 1D array-like of int
        second indices of V_m_l terms
    a_arr : list of sympy variables
        a_m_l coefficients of V_m_l terms, which multiplies cos(m*phi)*D_m_l
    b_arr : list of sympy variables
        b_m_l coefficients of V_m_l terms, which multiplies sin(m*phi)*D_m_l
    c_arr : list of sympy variables
        c_m_l coefficients of V_m_l terms, which multiplies cos(m*phi)*N_m_l-1
    d_arr : list of sympy variables
        d_m_l coefficients of V_m_l terms, which multiplies sin(m*phi)*N_m_l-1
    B0: float
        toroidal magnetic field strength scale, this is the strength of the
        1/R part of the magnetic field and is the Bphi at R=1.

    Returns
    -------
    value : sympy expression
        Sympy expression for the total dommaschk potential as a function
        of R, phi and Z.

    """
    value = B0 * phi  # phi term
    if len(ms) > 1:
        for i in range(len(ms)):
            value += V_m_l(
                R, phi, Z, ms[i], ls[i], a_arr[i], b_arr[i], c_arr[i], d_arr[i]
            )

    else:
        value += V_m_l(R, phi, Z, ms[0], ls[0], a_arr, b_arr, c_arr, d_arr)
    return value
