"""Classes for 2D surfaces embedded in 3D space."""

import numbers
import warnings

import numpy as np
from scipy.sparse.linalg import splu

from desc.backend import jnp, put, sign
from desc.basis import DoubleFiniteElementBasis, DoubleFourierSeries, ZernikePolynomial
from desc.io import InputReader
from desc.utils import copy_coeffs

from .core import Surface

__all__ = [
    "FourierRZToroidalSurface",
    "ZernikeRZToroidalSection",
    "convert_coefficients",
    "FiniteElementRZToroidalSurface",
    "TriangleFiniteElement",
]


class TriangleFiniteElement:
    """Class representing a triangle in a 2D grid of finite elements.

    Parameters
    ----------
    vertices: array-like, shape(3, 2)
        The three vertices of the triangle in (theta_i, zeta_i)
    K: integer
        The order of the finite elements to use, which gives (K+1)(K+2) / 2
        basis functions.
    """

    def __init__(self, vertices, K=1):
        self.vertices = vertices
        a1 = vertices[1, 0] * vertices[2, 1] - vertices[2, 0] * vertices[1, 1]
        a2 = vertices[2, 0] * vertices[0, 1] - vertices[0, 0] * vertices[2, 1]
        a3 = vertices[0, 0] * vertices[1, 1] - vertices[1, 0] * vertices[0, 1]
        self.a = np.array([a1, a2, a3])
        b1 = vertices[1, 1] - vertices[2, 1]
        b2 = vertices[2, 1] - vertices[0, 1]
        b3 = vertices[0, 1] - vertices[1, 1]
        self.b = np.array([b1, b2, b3])
        c1 = vertices[2, 0] - vertices[1, 0]
        c2 = vertices[0, 0] - vertices[2, 0]
        c3 = vertices[1, 0] - vertices[0, 0]
        self.c = np.array([c1, c2, c3])
        self.area = self.vertices[:, 0] @ self.b
        self.N_k = int((K + 1) * (K + 2) / 2)
        self.K = K

        # Going to construct equally spaced nodes for order K triangle,
        # which gives N_k such nodes.
        nodes = []

        # Start with the vertices of the triangle
        for i in range(3):
            nodes.append(vertices[i, :])

        # If K = 1, the vertices are the only nodes for basis functions
        if K > 1:
            # Add (K-1) equally spaced nodes on each triangle edge
            # for total of 3(K - 1) more nodes
            for i in range(3):
                for j in range(i + 1, 3):
                    for k in range(K - 1):
                        edge_node = (vertices[i, :] + vertices[j, :]) / K * (k + 1)
                        nodes.append(edge_node)

                # Fill in any nodes within the triangle by drawing rays between
                # the edge nodes that are more than 1 spacing away'
                if i == 0 and K > 2:
                    for k in range(1, K - 1):
                        edge_node1 = (vertices[i, :] + vertices[1, :]) / K * (k + 1)
                        edge_node2 = (vertices[i, :] + vertices[2, :]) / K * (k + 1)
                        center_node = (edge_node1 + edge_node2) / (K - 1) * k
                        nodes.append(center_node)

        self.nodes = nodes
        self.eta_nodes = self.get_barycentric_coordinates(self.nodes)
        assert nodes.shape[-1] == self.N_k

    def get_basis_functions(self, theta_zeta):
        """
        Gets the barycentric basis functions.

        Return the triangle basis functions, evaluated at the 2D theta
        and zeta mesh points provided to the function.

        Parameters
        ----------
        theta_zeta : 2D ndarray, shape (ntheta * nzeta, 2)
            Coordinates of the original grid, lying inside this triangle.

        Returns
        -------
        psi_q : (theta_zeta, N_k)

        """
        eta = self.get_barycentric_coordinates(theta_zeta)
        K = self.K
        basis_functions = np.zeros((theta_zeta.shape[0], self.N_k))
        q = 0
        for i in range(K):
            for j in range(K):
                for k in range(K):
                    if (i + j + k) == K:
                        basis_functions[:, q] = (
                            self.lagrange_polynomial(
                                eta[:, 0], self.eta_nodes[:, 0], i + 1, q
                            )
                            * self.lagrange_polynomial(
                                eta[:, 1], self.eta_nodes[:, 1], j + 1, q
                            )
                            * self.lagrange_polynomial(
                                eta[:, 2], self.eta_nodes[:, 2], k + 1, q
                            )
                        )
                        q += 1
        return basis_functions

    def lagrange_polynomial(self, eta_i, eta_nodes_i, order, q):
        """
        Computes lagrange polynomials.

        Computes the lagrange polynomial given the ith component of the
        Barycentric coordinates on a (theta, zeta) mesh, the ith component
        of the triangle nodes defined for the basis functions, the order
        of the polynomial, and the index q of which node this is.

        Parameters
        ----------
        eta_i : 1D ndarray, shape(ntheta * nzeta)
            The barycentric coordinate i defined at (theta, zeta) points.
        eta_nodes_i : 1D ndarray, shape(N_k)
            The barycentric coordinate i defined at the triangle nodes.
        order : integer
            Order of the polynomial.
        q : integer
            The index of the node we are using to define the basis function.
            Options are 0, ..., N_k - 1

        Returns
        -------
        lp : 1D ndarray, shape(ntheta * nzeta)
            The lagrange polynomial associated with the barycentric
            coordinate i, the polynomial order (order), and the node q.

        """
        denom = 1
        numerator = np.ones(len(eta_i))
        for i in range(len(eta_nodes_i)):
            denom *= eta_nodes_i[q] - eta_nodes_i[i]
            if i != q:
                numerator *= eta_i - eta_nodes_i[i]
        lp = numerator / denom
        return lp

    def get_barycentric_coordinates(self, theta_zeta):
        """
        Gets the barycentric coordinates, given a mesh in theta, zeta.

        Parameters
        ----------
        theta_zeta : 2D ndarray, shape (ntheta * nzeta, 2)
            Coordinates of the original grid, lying inside this triangle.

        Returns
        -------
        eta_u: 2D array, shape (ntheta * nzeta, 3)
            Barycentric coordinates defined by the triangle and evaluated
            at the points (theta, zeta).
        """
        # Get the Barycentric coordinates
        eta1 = self.a[0] + self.b[0] * theta_zeta + self.c[0] * theta_zeta
        eta2 = self.a[1] + self.b[1] * theta_zeta + self.c[1] * theta_zeta
        eta3 = self.a[2] + self.b[2] * theta_zeta + self.c[2] * theta_zeta
        eta = np.array([eta1, eta2, eta3]).reshape(-1, 3)

        # Check that all the points are indeed inside the triangle
        for i in range(eta.shape[0]):
            if eta1[i] < 0 or eta2[i] < 0 or eta3[i] < 0:
                warnings.warn(
                    "Found theta_zeta points outside the triangle ... "
                    "Not using these points to evaluate the barycentric "
                    "coordinates."
                )
            eta = np.delete(eta, i, 0)
        return eta


def convert_coefficients_2D(R_lmn, Z_lmn, R_basis, Z_basis, Rprime_basis, Zprime_basis):
    """Converts 2D Fourier to 2D FE representation.

    Parameters
    ----------
    R_lmn : ndarray, shape(k, 3)
        Fourier coefficients of R(1, theta, zeta)
    Z_lmn : ndarray, shape(k, 3)
        Fourier coefficients of Z(1, theta, zeta)
    modes_R : array-like, shape(k,2)
        poloidal and toroidal mode numbers [m,n] for R_lmn.
    modes_Z : array-like, shape(k,2)
        mode numbers associated with Z_lmn, defaults to modes_R
    R_basis : DoubleFourier
        Basis elements representing the dependence in (1, theta, phi).
    Z_basis : DoubleFourier
        Basis elements representing the dependence in (1, theta, phi).
    Rprime_basis : DoubleFiniteElementBasis
        Basis elements representing the dependence in (1, theta, phi).
    Zprime_basis : DoubleFiniteElementBasis
        Basis elements representing the dependence in (1, theta, phi).

    Returns
    -------
    tildeR_lmn : ndarray, shape (kk, 2)
        Finite element coefficients of R(1, theta, zeta)
    tildeZ_lmn : ndarray, shape (kk, 2)
        Finite element coefficients of Z(1, theta, zeta)

    """
    # Assume uniform grid
    rho = np.array([1.0])
    M = R_basis.M
    N = R_basis.N
    I = Rprime_basis.M
    J = Rprime_basis.N
    theta = np.linspace(0, 2 * np.pi, I)
    zeta = np.linspace(0, 2 * np.pi, J)
    nodes = np.array(np.meshgrid(rho, theta, zeta, indexing="ij"))
    Bjb_Z = np.zeros((I, J, I, J))
    Aj_Z = np.zeros((I, J))
    Bjb_R = np.zeros((I, J, I, J))
    Aj_R = np.zeros((I, J))
    for i in range(I):
        for j in range(J):
            for a in range(I):
                for b in range(J):
                    Bjb_Z[i, j, a, b] += np.sum(
                        np.sum(
                            Zprime_basis.evaluate(
                                nodes=nodes, modes=np.array([[0, i, j]])
                            )
                            * Zprime_basis.evaluate(
                                nodes=nodes, modes=np.array([[0, a, b]])
                            ),
                            axis=0,
                        )
                    )
                    Bjb_R[i, j, a, b] += np.sum(
                        np.sum(
                            Rprime_basis.evaluate(
                                nodes=nodes, modes=np.array([[0, i, j]])
                            )
                            * Rprime_basis.evaluate(
                                nodes=nodes, modes=np.array([[0, a, b]])
                            ),
                            axis=0,
                        )
                    )
    modes_R = R_basis.modes
    modes_Z = Z_basis.modes
    for i in range(I):
        for j in range(J):
            for m in range(M):
                for n in range(N):
                    # Sum over n, m and integrals over theta, zeta
                    Aj_Z[i, j] += Z_lmn[modes_Z[m + M * n, :]] * np.sum(
                        np.sum(
                            Z_basis.evaluate(nodes=nodes, modes=np.array([0, n, m]))
                            * Zprime_basis.evaluate(
                                nodes=nodes, modes=np.array([0, i, j])
                            ),
                            axis=0,
                        )
                    )
                    Aj_R[i, j] += R_lmn[modes_R[m + M * n, :]] * np.sum(
                        np.sum(
                            R_basis.evaluate(nodes=nodes, modes=np.array([0, n, m]))
                            * Rprime_basis.evaluate(
                                nodes=nodes, modes=np.array([0, i, j])
                            ),
                            axis=0,
                        )
                    )
    # Bjb and Aj should both be scaled by the grid spacing, but this cancels out
    # if the grid spacing is uniform, so we omit it here.
    Bjb_R = Bjb_R.reshape(I * J, I * J)
    Aj_R = Aj_R.reshape(I * J)
    Bjb_Z = Bjb_Z.reshape(I * J, I * J)
    Aj_Z = Aj_Z.reshape(I * J)

    # Constructed the matrices such that Bjb * Rprime = Aj and now need to solve
    # this linear system of equations. Use an LU
    lu = splu(Bjb_R)
    Rprime = lu.solve(Aj_R)
    Rprime_lmn = Rprime.reshape(I, J)
    lu = splu(Bjb_Z)
    Zprime = lu.solve(Aj_Z)
    Zprime_lmn = Zprime.reshape(I, J)
    return Rprime_lmn, Zprime_lmn


def convert_coefficients(
    R_lmn,
    Z_lmn,
    L_lmn,
    R_basis,
    Z_basis,
    L_basis,
    Rprime_basis,
    Zprime_basis,
    Lprime_basis,
):
    """Converts between double Fourier and double FE representation.

    Parameters
    ----------
    R_lmn : ndarray, shape(k, 3)
        Fourier coefficients of R(rho, theta, zeta)
    Z_lmn : ndarray, shape(k, 3)
        Fourier coefficients of Z(rho, theta, zeta)
    modes_R : array-like, shape(k,2)
        poloidal and toroidal mode numbers [m,n] for R_lmn.
    modes_Z : array-like, shape(k,2)
        mode numbers associated with Z_lmn, defaults to modes_R
    L_lmn : ndarray, shape(k, 3)
        Fourier coefficients of Lambda(rho, theta, zeta), Default = None
    R_basis : DoubleFourier or FourierZernike, shape (I, J) or shape (L, I, J)
        Basis elements representing the dependence in (rho, theta, phi).
    Z_basis : DoubleFourier or FourierZernike, shape (I, J) or shape (L, I, J)
        Basis elements representing the dependence in (rho, theta, phi).
    L_basis : DoubleFourier or FourierZernike, shape (I, J) or shape (L, I, J)
        Basis elements representing the dependence in (rho, theta, phi).
    J : int
        Number of FE modes to use in toroidal direction.

    Returns
    -------
    tildeR_lmn : ndarray, shape (kk, 3)
        Finite element coefficients of R(rho, theta, zeta)
    tildeZ_lmn : ndarray, shape (kk, 3)
        Finite element coefficients of Z(rho, theta, zeta)
    tildeL_lmn : ndarray, shape (kk, 3)
        Finite element coefficients of L(rho, theta, zeta)

    """
    # Assume uniform grid
    rho = np.array([1.0])
    M = R_basis.M
    N = R_basis.N
    I = max(Rprime_basis.M, 1)
    J = max(Rprime_basis.N, 1)
    L = max(Rprime_basis.L, 1)
    theta = np.linspace(0, 2 * np.pi, I)
    zeta = np.linspace(0, 2 * np.pi, J)
    nodes = np.array(np.meshgrid(rho, theta, zeta, indexing="ij")).reshape(3, I * J).T
    Bjb_Z = np.zeros((I, J, I, J))
    Aj_Z = np.zeros((I, J, L))
    Bjb_R = np.zeros((I, J, I, J))
    Aj_R = np.zeros((I, J, L))
    Bjb_L = np.zeros((I, J, I, J))
    Aj_L = np.zeros((I, J, L))
    for i in range(I):
        for j in range(J):
            for a in range(I):
                for b in range(J):
                    Bjb_Z[i, j, a, b] += np.sum(
                        np.sum(
                            Zprime_basis.evaluate(
                                nodes=nodes, modes=np.array([[0, i, j]])
                            )
                            * Zprime_basis.evaluate(
                                nodes=nodes, modes=np.array([[0, a, b]])
                            ),
                            axis=0,
                        )
                    )
                    Bjb_R[i, j, a, b] += np.sum(
                        np.sum(
                            Rprime_basis.evaluate(
                                nodes=nodes, modes=np.array([[0, i, j]])
                            )
                            * Rprime_basis.evaluate(
                                nodes=nodes, modes=np.array([[0, a, b]])
                            ),
                            axis=0,
                        )
                    )
    rho = np.linspace(0, 1.0, L, endpoint=True)
    nodes = (
        np.array(np.meshgrid(rho, theta, zeta, indexing="ij")).reshape(3, I * J * L).T
    )
    modes_R = R_basis.modes
    modes_Z = Z_basis.modes
    modes_L = L_basis.modes
    for i in range(I):
        for j in range(J):
            for k in range(L):
                for n in range(M):
                    for m in range(N):
                        for l in range(L):
                            # Sum over n, m, l and integrals over theta, zeta, rho
                            Aj_Z[i, j, k] += (
                                Z_lmn[modes_Z[l + L * m + L * M * n, :]]
                                * np.sum(
                                    np.sum(
                                        np.sum(
                                            Z_basis.evaluate(
                                                nodes=nodes,
                                                modes=modes_Z[l + L * m + L * M * n, :],
                                            )
                                            * Zprime_basis.evaluate(
                                                nodes=nodes, modes=np.array([0, i, j])
                                            ),
                                            axis=0,
                                        ),
                                        axis=0,
                                    )
                                )
                                / (k + 1)
                            )
                            Aj_R[i, j, k] += (
                                R_lmn[modes_R[l + L * m + L * M * n, :]]
                                * np.sum(
                                    np.sum(
                                        np.sum(
                                            R_basis.evaluate(
                                                nodes=nodes,
                                                modes=modes_R[l + L * m + L * M * n, :],
                                            )
                                            * Rprime_basis.evaluate(
                                                nodes=nodes, modes=np.array([0, i, j])
                                            ),
                                            axis=0,
                                        ),
                                        axis=0,
                                    )
                                )
                                / (k + 1)
                            )
                            Aj_L[i, j, k] += (
                                L_lmn[modes_L[l + L * m + L * M * n, :]]
                                * np.sum(
                                    np.sum(
                                        np.sum(
                                            L_basis.evaluate(
                                                nodes=nodes,
                                                modes=modes_L[l + L * m + L * M * n, :],
                                            )
                                            * Lprime_basis.evaluate(
                                                nodes=nodes, modes=np.array([0, i, j])
                                            ),
                                            axis=0,
                                        ),
                                        axis=0,
                                    )
                                )
                                / (k + 1)
                            )
    # Bjb and Aj should both be scaled by the grid spacing, but this cancels out
    # if the grid spacing is uniform, so we omit it here.
    # However, factor of pi from the orthonormality of the radial basis functions
    # being used in the finite element representation.
    Bjb_R = Bjb_R.reshape(I * J, I * J)
    print(Bjb_R)
    Bjb_R_expanded = np.zeros((I * J, L, I * J, L))
    for ii in range(L):
        Bjb_R_expanded[:, ii, :, ii] = Bjb_R
    Bjb_R = np.reshape(Bjb_R_expanded, (I * J * L, I * J * L))
    print(Bjb_R.shape)
    Aj_R = Aj_R.reshape(I * J * L) * np.pi
    Bjb_Z = Bjb_Z.reshape(I * J, I * J)
    Bjb_Z = np.tile(Bjb_Z, L)  # repeat this matrix L times for the linear solve
    Aj_Z = Aj_Z.reshape(I * J * L) * np.pi
    Bjb_L = Bjb_L.reshape(I * J, I * J)
    Bjb_L = np.tile(Bjb_L, L)  # repeat this matrix L times for the linear solve
    Aj_L = Aj_L.reshape(I * J * L) * np.pi

    # Constructed the matrices such that Bjb * Rprime = Aj and now need to solve
    # this linear system of equations. Use an LU
    lu = splu(Bjb_R)
    Rprime = lu.solve(Aj_R)
    Rprime_lmn = Rprime.reshape(L * I * J)
    lu = splu(Bjb_Z)
    Zprime = lu.solve(Aj_Z)
    Zprime_lmn = Zprime.reshape(L * I * J)
    lu = splu(Bjb_L)
    Lprime = lu.solve(Aj_L)
    Lprime_lmn = Lprime.reshape(L * I * J)
    return Rprime_lmn, Zprime_lmn, Lprime_lmn


class FiniteElementRZToroidalSurface(Surface):
    """Toroidal surface represented by finite elements in poloidal and toroidal angles.

    Parameters
    ----------
    R_lmn, Z_lmn : array-like, shape(k,)
        Finite Element coefficients for R and Z in cylindrical coordinates
    modes_R : array-like, shape(k,2)
        poloidal and toroidal mode numbers [m,n] for R_lmn
    modes_Z : array-like, shape(k,2)
        mode numbers associated with Z_lmn, defaults to modes_R
    rho : float [0,1]
        flux surface label for the toroidal surface
    name : str
        name for this surface
    check_orientation : bool
        ensure that this surface has a right handed orientation. Do not set to False
        unless you are sure the parameterization you have given is right handed
        (ie, e_theta x e_zeta points outward from the surface).

    """

    _io_attrs_ = Surface._io_attrs_ + [
        "_R_lmn",
        "_Z_lmn",
        "_R_basis",
        "_Z_basis",
        "rho",
        "_NFP",
    ]

    def __init__(
        self,
        R_lmn=None,
        Z_lmn=None,
        modes_R=None,
        modes_Z=None,
        NFP=1,
        sym="auto",
        rho=1,
        name="",
        check_orientation=True,
    ):
        self.fs = FourierRZToroidalSurface(
            R_lmn=R_lmn,
            Z_lmn=Z_lmn,
            modes_R=modes_R,
            modes_Z=modes_Z,
            NFP=NFP,
            sym=sym,
            rho=rho,
            name=name,
            check_orientation=check_orientation,
        )
        self._R_basis = DoubleFiniteElementBasis(M=32, N=32)
        self._Z_basis = DoubleFiniteElementBasis(M=32, N=32)
        R_lmn, Z_lmn = convert_coefficients_2D(
            self.fs.R_lmn,
            self.fs.Z_lmn,
            self.fs.R_basis,
            self.fs.Z_basis,
            self._R_basis,
            self._Z_basis,
        )
        I = self.fs._M * 2
        J = self.fs._N * 2
        modes_R, modes_Z = np.meshgrid(I, J, indexing="ij")
        modes_R = modes_R.reshape(-1, 2)
        modes_Z = modes_Z.reshape(-1, 2)
        self.R_lmn = R_lmn
        self.Z_lmn = Z_lmn
        self.rho = rho
        self._R_lmn = copy_coeffs(R_lmn, modes_R, self.R_basis.modes[:, 1:])
        self._Z_lmn = copy_coeffs(Z_lmn, modes_Z, self.Z_basis.modes[:, 1:])

    @property
    def NFP(self):
        """int: Number of (toroidal) field periods."""
        return self.fs._NFP

    @NFP.setter
    def NFP(self, new):
        assert (
            isinstance(new, numbers.Real) and int(new) == new and new > 0
        ), f"NFP should be a positive integer, got {type(new)}"
        self.fs.change_resolution(NFP=new)

    @property
    def R_basis(self):
        """Double finite element basis for R."""
        return self._R_basis

    @property
    def Z_basis(self):
        """Double finite element basis for Z."""
        return self._Z_basis

    def change_resolution(self, *args, **kwargs):
        """Change the maximum poloidal and toroidal resolution."""
        assert (
            ((len(args) in [2, 3]) and len(kwargs) == 0)
            or ((len(args) in [2, 3]) and len(kwargs) in [1, 2])
            or (len(args) == 0)
        ), (
            "change_resolution should be called with 2 (M,N) or 3 (L,M,N) "
            + "positional arguments or only keyword arguments."
        )
        L = kwargs.pop("L", None)
        M = kwargs.pop("M", None)
        N = kwargs.pop("N", None)
        NFP = kwargs.pop("NFP", None)
        sym = kwargs.pop("sym", None)
        assert len(kwargs) == 0, "change_resolution got unexpected kwarg: {kwargs}"
        self._NFP = NFP if NFP is not None else self.NFP
        self._sym = sym if sym is not None else self.sym
        if L is not None:
            warnings.warn(
                "FourierRZToroidalSurface does not have radial resolution, ignoring L"
            )
        if len(args) == 2:
            M, N = args
        elif len(args) == 3:
            L, M, N = args

        if (
            ((N is not None) and (N != self.N))
            or ((M is not None) and (M != self.M))
            or (NFP is not None)
        ):
            M = M if M is not None else self.M
            N = N if N is not None else self.N
            R_modes_old = self.R_basis.modes
            Z_modes_old = self.Z_basis.modes
            self.R_basis.change_resolution(I=M, J=N)
            self.Z_basis.change_resolution(I=M, J=N)
            self.R_lmn = copy_coeffs(self.R_lmn, R_modes_old, self.R_basis.modes)
            self.Z_lmn = copy_coeffs(self.Z_lmn, Z_modes_old, self.Z_basis.modes)
            self._M = M
            self._N = N

    @property
    def R_lmn(self):
        """ndarray: Spectral coefficients for R."""
        return self._R_lmn

    @R_lmn.setter
    def R_lmn(self, new):
        if len(new) == self.R_basis.num_modes:
            self._R_lmn = jnp.asarray(new)
        else:
            raise ValueError(
                f"R_lmn should have the same size as the basis, got {len(new)} for "
                + f"basis with {self.R_basis.num_modes} modes."
            )

    @property
    def Z_lmn(self):
        """ndarray: Spectral coefficients for Z."""
        return self._Z_lmn

    @Z_lmn.setter
    def Z_lmn(self, new):
        if len(new) == self.Z_basis.num_modes:
            self._Z_lmn = jnp.asarray(new)
        else:
            raise ValueError(
                f"Z_lmn should have the same size as the basis, got {len(new)} for "
                + f"basis with {self.R_basis.num_modes} modes."
            )

    def get_coeffs(self, m, n=0):
        """Get finite element coefficients for given mode number(s)."""
        n = np.atleast_1d(n).astype(int)
        m = np.atleast_1d(m).astype(int)

        m, n = np.broadcast_arrays(m, n)
        R = np.zeros_like(m).astype(float)
        Z = np.zeros_like(m).astype(float)

        mn = np.array([m, n]).T
        idxR = np.where(
            (mn[:, np.newaxis, :] == self.R_basis.modes[np.newaxis, :, 1:]).all(axis=-1)
        )
        idxZ = np.where(
            (mn[:, np.newaxis, :] == self.Z_basis.modes[np.newaxis, :, 1:]).all(axis=-1)
        )

        R[idxR[0]] = self.R_lmn[idxR[1]]
        Z[idxZ[0]] = self.Z_lmn[idxZ[1]]
        return R, Z

    def set_coeffs(self, m, n=0, R=None, Z=None):
        """Set specific finite element coefficients."""
        m, n, R, Z = (
            np.atleast_1d(m),
            np.atleast_1d(n),
            np.atleast_1d(R),
            np.atleast_1d(Z),
        )
        m, n, R, Z = np.broadcast_arrays(m, n, R, Z)
        for mm, nn, RR, ZZ in zip(m, n, R, Z):
            if RR is not None:
                idxR = self.R_basis.get_idx(0, mm, nn)
                self.R_lmn = put(self.R_lmn, idxR, RR)
            if ZZ is not None:
                idxZ = self.Z_basis.get_idx(0, mm, nn)
                self.Z_lmn = put(self.Z_lmn, idxZ, ZZ)

    @classmethod
    def from_input_file(cls, path):
        """Create a finite element surface from Fourier coefficients in a file.

        Parameters
        ----------
        path : Path-like or str
            Path to DESC or VMEC input file.

        Returns
        -------
        surface : FiniteElementRZToroidalSurface
            Surface with given finite element coefficients.

        """
        surf = cls()
        fs = surf.fs.from_input_file(path=path)
        R_lmn, Z_lmn = convert_coefficients_2D(
            fs.R_lmn,
            fs.Z_lmn,
            fs.R_basis,
            fs.Z_basis,
            surf._R_basis,
            surf._Z_basis,
        )
        I = fs._M * 2
        J = fs._N * 2
        modes_R, modes_Z = np.meshgrid(I, J, indexing="ij")
        modes_R = modes_R.reshape(-1, 2)
        modes_Z = modes_Z.reshape(-1, 2)
        surf = cls(R_lmn=R_lmn, Z_lmn=Z_lmn, modes_R=modes_R, modes_Z=modes_Z)
        return surf

    @classmethod
    def from_near_axis(cls, aspect_ratio, elongation, mirror_ratio, axis_Z, NFP=1):
        """Create a surface from a near-axis model for quasi-poloidal/quasi-isodynamic.

        Parameters
        ----------
        aspect_ratio : float
            Aspect ratio of the geometry = major radius / average cross-sectional area.
        elongation : float
            Elongation of the elliptical surface = major axis / minor axis.
        mirror_ratio : float
            Mirror ratio generated by toroidal variation of the cross-sectional area.
            Must be < 2.
        axis_Z : float
            Vertical extent of the magnetic axis Z coordinate.
            Coefficient of sin(2*phi).
        NFP : int
            Number of field periods.

        Returns
        -------
        surface : FourierRZToroidalSurface
            Surface with given geometric properties.

        """
        surf = cls()
        fs = surf.fs.from_near_axis(aspect_ratio, elongation, mirror_ratio, axis_Z, NFP)
        R_lmn, Z_lmn = convert_coefficients_2D(
            fs.R_lmn,
            fs.Z_lmn,
            fs.R_basis,
            fs.Z_basis,
            surf._R_basis,
            surf._Z_basis,
        )
        I = fs._M * 2
        J = fs._N * 2
        modes_R, modes_Z = np.meshgrid(I, J, indexing="ij")
        modes_R = modes_R.reshape(-1, 2)
        modes_Z = modes_Z.reshape(-1, 2)
        surf = cls(R_lmn=R_lmn, Z_lmn=Z_lmn, modes_R=modes_R, modes_Z=modes_Z)
        return surf


class FourierRZToroidalSurface(Surface):
    """Toroidal surface represented by Fourier series in poloidal and toroidal angles.

    Parameters
    ----------
    R_lmn, Z_lmn : array-like, shape(k,)
        Fourier coefficients for R and Z in cylindrical coordinates
    modes_R : array-like, shape(k,2)
        poloidal and toroidal mode numbers [m,n] for R_lmn.
    modes_Z : array-like, shape(k,2)
        mode numbers associated with Z_lmn, defaults to modes_R
    NFP : int
        number of field periods
    sym : bool
        whether to enforce stellarator symmetry. Default is "auto" which enforces if
        modes are symmetric. If True, non-symmetric modes will be truncated.
    rho : float [0,1]
        flux surface label for the toroidal surface
    name : str
        name for this surface
    check_orientation : bool
        ensure that this surface has a right handed orientation. Do not set to False
        unless you are sure the parameterization you have given is right handed
        (ie, e_theta x e_zeta points outward from the surface).

    """

    _io_attrs_ = Surface._io_attrs_ + [
        "_R_lmn",
        "_Z_lmn",
        "_R_basis",
        "_Z_basis",
        "rho",
        "_NFP",
    ]

    def __init__(
        self,
        R_lmn=None,
        Z_lmn=None,
        modes_R=None,
        modes_Z=None,
        NFP=1,
        sym="auto",
        rho=1,
        name="",
        check_orientation=True,
    ):
        if R_lmn is None:
            R_lmn = np.array([10, 1])
            modes_R = np.array([[0, 0], [1, 0]])
        if Z_lmn is None:
            Z_lmn = np.array([0, -1])
            modes_Z = np.array([[0, 0], [-1, 0]])
        if modes_Z is None:
            modes_Z = modes_R
        R_lmn, Z_lmn, modes_R, modes_Z = map(
            np.asarray, (R_lmn, Z_lmn, modes_R, modes_Z)
        )

        assert issubclass(modes_R.dtype.type, np.integer)
        assert issubclass(modes_Z.dtype.type, np.integer)

        MR = np.max(abs(modes_R[:, 0]))
        NR = np.max(abs(modes_R[:, 1]))
        MZ = np.max(abs(modes_Z[:, 0]))
        NZ = np.max(abs(modes_Z[:, 1]))
        self._L = 0
        self._M = max(MR, MZ)
        self._N = max(NR, NZ)
        if sym == "auto":
            if np.all(
                R_lmn[np.where(sign(modes_R[:, 0]) != sign(modes_R[:, 1]))] == 0
            ) and np.all(
                Z_lmn[np.where(sign(modes_Z[:, 0]) == sign(modes_Z[:, 1]))] == 0
            ):
                sym = True
            else:
                sym = False

        self._R_basis = DoubleFourierSeries(
            M=MR, N=NR, NFP=NFP, sym="cos" if sym else False
        )
        self._Z_basis = DoubleFourierSeries(
            M=MZ, N=NZ, NFP=NFP, sym="sin" if sym else False
        )

        self._R_lmn = copy_coeffs(R_lmn, modes_R, self.R_basis.modes[:, 1:])
        self._Z_lmn = copy_coeffs(Z_lmn, modes_Z, self.Z_basis.modes[:, 1:])
        self._NFP = NFP
        self._sym = sym
        self.rho = rho

        if check_orientation and self._compute_orientation() == -1:
            warnings.warn(
                "Left handed coordinates detected, switching sign of theta."
                + " To avoid this warning in the future, switch the sign of all"
                + " modes with m<0"
            )
            self._flip_orientation()
            assert self._compute_orientation() == 1

        self.name = name

    @property
    def NFP(self):
        """int: Number of (toroidal) field periods."""
        return self._NFP

    @NFP.setter
    def NFP(self, new):
        assert (
            isinstance(new, numbers.Real) and int(new) == new and new > 0
        ), f"NFP should be a positive integer, got {type(new)}"
        self.change_resolution(NFP=new)

    @property
    def R_basis(self):
        """DoubleFourierSeries: Spectral basis for R."""
        return self._R_basis

    @property
    def Z_basis(self):
        """DoubleFourierSeries: Spectral basis for Z."""
        return self._Z_basis

    def change_resolution(self, *args, **kwargs):
        """Change the maximum poloidal and toroidal resolution."""
        assert (
            ((len(args) in [2, 3]) and len(kwargs) == 0)
            or ((len(args) in [2, 3]) and len(kwargs) in [1, 2])
            or (len(args) == 0)
        ), (
            "change_resolution should be called with 2 (M,N) or 3 (L,M,N) "
            + "positional arguments or only keyword arguments."
        )
        L = kwargs.pop("L", None)
        M = kwargs.pop("M", None)
        N = kwargs.pop("N", None)
        NFP = kwargs.pop("NFP", None)
        sym = kwargs.pop("sym", None)
        assert len(kwargs) == 0, "change_resolution got unexpected kwarg: {kwargs}"
        self._NFP = NFP if NFP is not None else self.NFP
        self._sym = sym if sym is not None else self.sym
        if L is not None:
            warnings.warn(
                "FourierRZToroidalSurface does not have radial resolution, ignoring L"
            )
        if len(args) == 2:
            M, N = args
        elif len(args) == 3:
            L, M, N = args

        if (
            ((N is not None) and (N != self.N))
            or ((M is not None) and (M != self.M))
            or (NFP is not None)
        ):
            M = M if M is not None else self.M
            N = N if N is not None else self.N
            R_modes_old = self.R_basis.modes
            Z_modes_old = self.Z_basis.modes
            self.R_basis.change_resolution(
                M=M, N=N, NFP=self.NFP, sym="cos" if self.sym else self.sym
            )
            self.Z_basis.change_resolution(
                M=M, N=N, NFP=self.NFP, sym="sin" if self.sym else self.sym
            )
            self.R_lmn = copy_coeffs(self.R_lmn, R_modes_old, self.R_basis.modes)
            self.Z_lmn = copy_coeffs(self.Z_lmn, Z_modes_old, self.Z_basis.modes)
            self._M = M
            self._N = N

    @property
    def R_lmn(self):
        """ndarray: Spectral coefficients for R."""
        return self._R_lmn

    @R_lmn.setter
    def R_lmn(self, new):
        if len(new) == self.R_basis.num_modes:
            self._R_lmn = jnp.asarray(new)
        else:
            raise ValueError(
                f"R_lmn should have the same size as the basis, got {len(new)} for "
                + f"basis with {self.R_basis.num_modes} modes."
            )

    @property
    def Z_lmn(self):
        """ndarray: Spectral coefficients for Z."""
        return self._Z_lmn

    @Z_lmn.setter
    def Z_lmn(self, new):
        if len(new) == self.Z_basis.num_modes:
            self._Z_lmn = jnp.asarray(new)
        else:
            raise ValueError(
                f"Z_lmn should have the same size as the basis, got {len(new)} for "
                + f"basis with {self.R_basis.num_modes} modes."
            )

    def get_coeffs(self, m, n=0):
        """Get Fourier coefficients for given mode number(s)."""
        n = np.atleast_1d(n).astype(int)
        m = np.atleast_1d(m).astype(int)

        m, n = np.broadcast_arrays(m, n)
        R = np.zeros_like(m).astype(float)
        Z = np.zeros_like(m).astype(float)

        mn = np.array([m, n]).T
        idxR = np.where(
            (mn[:, np.newaxis, :] == self.R_basis.modes[np.newaxis, :, 1:]).all(axis=-1)
        )
        idxZ = np.where(
            (mn[:, np.newaxis, :] == self.Z_basis.modes[np.newaxis, :, 1:]).all(axis=-1)
        )

        R[idxR[0]] = self.R_lmn[idxR[1]]
        Z[idxZ[0]] = self.Z_lmn[idxZ[1]]
        return R, Z

    def set_coeffs(self, m, n=0, R=None, Z=None):
        """Set specific Fourier coefficients."""
        m, n, R, Z = (
            np.atleast_1d(m),
            np.atleast_1d(n),
            np.atleast_1d(R),
            np.atleast_1d(Z),
        )
        m, n, R, Z = np.broadcast_arrays(m, n, R, Z)
        for mm, nn, RR, ZZ in zip(m, n, R, Z):
            if RR is not None:
                idxR = self.R_basis.get_idx(0, mm, nn)
                self.R_lmn = put(self.R_lmn, idxR, RR)
            if ZZ is not None:
                idxZ = self.Z_basis.get_idx(0, mm, nn)
                self.Z_lmn = put(self.Z_lmn, idxZ, ZZ)

    @classmethod
    def from_input_file(cls, path):
        """Create a surface from Fourier coefficients in a DESC or VMEC input file.

        Parameters
        ----------
        path : Path-like or str
            Path to DESC or VMEC input file.

        Returns
        -------
        surface : FourierRZToroidalSurface
            Surface with given Fourier coefficients.

        """
        f = open(path)
        if "&INDATA" in f.readlines()[0].upper():  # vmec input, convert to desc
            inputs = InputReader.parse_vmec_inputs(f)[-1]
        else:
            inputs = InputReader().parse_inputs(f)[-1]
        if (inputs["bdry_ratio"] is not None) and (inputs["bdry_ratio"] != 1):
            warnings.warn(
                "boundary_ratio = {} != 1, surface may not be as expected".format(
                    inputs["bdry_ratio"]
                )
            )
        surf = cls(
            inputs["surface"][:, 3],
            inputs["surface"][:, 4],
            inputs["surface"][:, 1:3].astype(int),
            inputs["surface"][:, 1:3].astype(int),
            inputs["NFP"],
            inputs["sym"],
        )
        return surf

    @classmethod
    def from_near_axis(cls, aspect_ratio, elongation, mirror_ratio, axis_Z, NFP=1):
        """Create a surface from a near-axis model for quasi-poloidal/quasi-isodynamic.

        Parameters
        ----------
        aspect_ratio : float
            Aspect ratio of the geometry = major radius / average cross-sectional area.
        elongation : float
            Elongation of the elliptical surface = major axis / minor axis.
        mirror_ratio : float
            Mirror ratio generated by toroidal variation of the cross-sectional area.
            Must be < 2.
        axis_Z : float
            Vertical extent of the magnetic axis Z coordinate.
            Coefficient of sin(2*phi).
        NFP : int
            Number of field periods.

        Returns
        -------
        surface : FourierRZToroidalSurface
            Surface with given geometric properties.

        """
        assert mirror_ratio <= 2
        a = np.sqrt(elongation) / aspect_ratio  # major axis
        b = 1 / (aspect_ratio * np.sqrt(elongation))  # minor axis
        epsilon = (2 - np.sqrt(4 - mirror_ratio**2)) / mirror_ratio

        R_lmn = np.array(
            [
                1,
                (elongation + 1) * b / 2,
                -1 / 5,
                a * epsilon,
                (elongation - 1) * b / 2,
                (elongation - 1) * b / 2,
            ]
        )
        Z_lmn = np.array(
            [
                -(elongation + 1) * b / 2,
                axis_Z,
                -b * epsilon,
                -(elongation - 1) * b / 2,
                (elongation - 1) * b / 2,
            ]
        )
        modes_R = np.array([[0, 0], [1, 0], [0, 2], [1, 1], [1, 2], [-1, -2]])
        modes_Z = np.array([[-1, 0], [0, -2], [-1, 1], [1, -2], [-1, 2]])

        surf = cls(R_lmn=R_lmn, Z_lmn=Z_lmn, modes_R=modes_R, modes_Z=modes_Z, NFP=NFP)
        return surf


class ZernikeRZToroidalSection(Surface):
    """A toroidal cross section represented by a Zernike polynomial in R,Z.

    Parameters
    ----------
    R_lmn, Z_lmn : array-like, shape(k,)
        zernike coefficients
    modes_R : array-like, shape(k,2)
        radial and poloidal mode numbers [l,m] for R_lmn
    modes_Z : array-like, shape(k,2)
        radial and poloidal mode numbers [l,m] for Z_lmn. If None defaults to modes_R.
    sym : bool
        whether to enforce stellarator symmetry. Default is "auto" which enforces if
        modes are symmetric. If True, non-symmetric modes will be truncated.
    spectral_indexing : {``'ansi'``, ``'fringe'``}
        Indexing method, default value = ``'ansi'``

        For L=0, all methods are equivalent and give a "chevron" shaped
        basis (only the outer edge of the zernike pyramid of width M).
        For L>0, the indexing scheme defines order of the basis functions:

        ``'ansi'``: ANSI indexing fills in the pyramid with triangles of
        decreasing size, ending in a triangle shape. For L == M,
        the traditional ANSI pyramid indexing is recovered. For L>M, adds rows
        to the bottom of the pyramid, increasing L while keeping M constant,
        giving a "house" shape

        ``'fringe'``: Fringe indexing fills in the pyramid with chevrons of
        decreasing size, ending in a diamond shape for L=2*M where
        the traditional fringe/U of Arizona indexing is recovered.
        For L > 2*M, adds chevrons to the bottom, making a hexagonal diamond
    zeta : float [0,2pi)
        toroidal angle for the section.
    name : str
        name for this surface
    check_orientation : bool
        ensure that this surface has a right handed orientation. Do not set to False
        unless you are sure the parameterization you have given is right handed
        (ie, e_theta x e_zeta points outward from the surface).

    """

    _io_attrs_ = Surface._io_attrs_ + [
        "_R_lmn",
        "_Z_lmn",
        "_R_basis",
        "_Z_basis",
        "zeta",
        "_spectral_indexing",
    ]

    def __init__(
        self,
        R_lmn=None,
        Z_lmn=None,
        modes_R=None,
        modes_Z=None,
        spectral_indexing="ansi",
        sym="auto",
        zeta=0.0,
        name="",
        check_orientation=True,
    ):
        if R_lmn is None:
            R_lmn = np.array([10, 1])
            modes_R = np.array([[0, 0], [1, 1]])
        if Z_lmn is None:
            Z_lmn = np.array([0, -1])
            modes_Z = np.array([[0, 0], [1, -1]])
        if modes_Z is None:
            modes_Z = modes_R
        R_lmn, Z_lmn, modes_R, modes_Z = map(
            np.asarray, (R_lmn, Z_lmn, modes_R, modes_Z)
        )

        assert issubclass(modes_R.dtype.type, np.integer)
        assert issubclass(modes_Z.dtype.type, np.integer)

        LR = np.max(abs(modes_R[:, 0]))
        MR = np.max(abs(modes_R[:, 1]))
        LZ = np.max(abs(modes_Z[:, 0]))
        MZ = np.max(abs(modes_Z[:, 1]))
        self._L = max(LR, LZ)
        self._M = max(MR, MZ)
        self._N = 0

        if sym == "auto":
            if np.all(
                R_lmn[np.where(sign(modes_R[:, 0]) != sign(modes_R[:, 1]))] == 0
            ) and np.all(
                Z_lmn[np.where(sign(modes_Z[:, 0]) == sign(modes_Z[:, 1]))] == 0
            ):
                sym = True
            else:
                sym = False

        self._R_basis = ZernikePolynomial(
            L=max(LR, MR),
            M=max(LR, MR),
            spectral_indexing=spectral_indexing,
            sym="cos" if sym else False,
        )
        self._Z_basis = ZernikePolynomial(
            L=max(LZ, MZ),
            M=max(LZ, MZ),
            spectral_indexing=spectral_indexing,
            sym="sin" if sym else False,
        )

        self._R_lmn = copy_coeffs(R_lmn, modes_R, self.R_basis.modes[:, :2])
        self._Z_lmn = copy_coeffs(Z_lmn, modes_Z, self.Z_basis.modes[:, :2])
        self._sym = sym
        self._spectral_indexing = spectral_indexing

        self.zeta = zeta

        if check_orientation and self._compute_orientation() == -1:
            warnings.warn(
                "Left handed coordinates detected, switching sign of theta."
                + " To avoid this warning in the future, switch the sign of all"
                + " modes with m<0"
            )
            self._flip_orientation()
            assert self._compute_orientation() == 1

        self.name = name

    @property
    def spectral_indexing(self):
        """str: Type of spectral indexing for Zernike basis."""
        return self._spectral_indexing

    @property
    def R_basis(self):
        """ZernikePolynomial: Spectral basis for R."""
        return self._R_basis

    @property
    def Z_basis(self):
        """ZernikePolynomial: Spectral basis for Z."""
        return self._Z_basis

    def change_resolution(self, *args, **kwargs):
        """Change the maximum radial and poloidal resolution."""
        assert (
            ((len(args) in [2, 3]) and len(kwargs) == 0)
            or ((len(args) in [2, 3]) and len(kwargs) in [1, 2])
            or (len(args) == 0)
        ), (
            "change_resolution should be called with 2 (M,N) or 3 (L,M,N) "
            + "positional arguments or only keyword arguments."
        )
        L = kwargs.pop("L", None)
        M = kwargs.pop("M", None)
        N = kwargs.pop("N", None)
        sym = kwargs.pop("sym", None)
        assert len(kwargs) == 0, "change_resolution got unexpected kwarg: {kwargs}"
        self._sym = sym if sym is not None else self.sym
        if N is not None:
            warnings.warn(
                "ZernikeRZToroidalSection does not have toroidal resolution, ignoring N"
            )
        if len(args) == 2:
            L, M = args
        elif len(args) == 3:
            L, M, N = args

        if ((L is not None) and (L != self.L)) or ((M is not None) and (M != self.M)):
            L = L if L is not None else self.L
            M = M if M is not None else self.M
            R_modes_old = self.R_basis.modes
            Z_modes_old = self.Z_basis.modes
            self.R_basis.change_resolution(
                L=L, M=M, sym="cos" if self.sym else self.sym
            )
            self.Z_basis.change_resolution(
                L=L, M=M, sym="sin" if self.sym else self.sym
            )
            self.R_lmn = copy_coeffs(self.R_lmn, R_modes_old, self.R_basis.modes)
            self.Z_lmn = copy_coeffs(self.Z_lmn, Z_modes_old, self.Z_basis.modes)
            self._L = L
            self._M = M

    @property
    def R_lmn(self):
        """ndarray: Spectral coefficients for R."""
        return self._R_lmn

    @R_lmn.setter
    def R_lmn(self, new):
        if len(new) == self.R_basis.num_modes:
            self._R_lmn = jnp.asarray(new)
        else:
            raise ValueError(
                f"R_lmn should have the same size as the basis, got {len(new)} for "
                + f"basis with {self.R_basis.num_modes} modes."
            )

    @property
    def Z_lmn(self):
        """ndarray: Spectral coefficients for Z."""
        return self._Z_lmn

    @Z_lmn.setter
    def Z_lmn(self, new):
        if len(new) == self.Z_basis.num_modes:
            self._Z_lmn = jnp.asarray(new)
        else:
            raise ValueError(
                f"Z_lmn should have the same size as the basis, got {len(new)} for "
                + f"basis with {self.R_basis.num_modes} modes."
            )

    def get_coeffs(self, l, m=0):
        """Get Zernike coefficients for given mode number(s)."""
        l = np.atleast_1d(l).astype(int)
        m = np.atleast_1d(m).astype(int)

        l, m = np.broadcast_arrays(l, m)
        R = np.zeros_like(m).astype(float)
        Z = np.zeros_like(m).astype(float)

        lm = np.array([l, m]).T
        idxR = np.where(
            (lm[:, np.newaxis, :] == self.R_basis.modes[np.newaxis, :, :2]).all(axis=-1)
        )
        idxZ = np.where(
            (lm[:, np.newaxis, :] == self.Z_basis.modes[np.newaxis, :, :2]).all(axis=-1)
        )

        R[idxR[0]] = self.R_lmn[idxR[1]]
        Z[idxZ[0]] = self.Z_lmn[idxZ[1]]
        return R, Z

    def set_coeffs(self, l, m=0, R=None, Z=None):
        """Set specific Zernike coefficients."""
        l, m, R, Z = (
            np.atleast_1d(l),
            np.atleast_1d(m),
            np.atleast_1d(R),
            np.atleast_1d(Z),
        )
        l, m, R, Z = np.broadcast_arrays(l, m, R, Z)
        for ll, mm, RR, ZZ in zip(l, m, R, Z):
            if RR is not None:
                idxR = self.R_basis.get_idx(ll, mm, 0)
                self.R_lmn = put(self.R_lmn, idxR, RR)
            if ZZ is not None:
                idxZ = self.Z_basis.get_idx(ll, mm, 0)
                self.Z_lmn = put(self.Z_lmn, idxZ, ZZ)
