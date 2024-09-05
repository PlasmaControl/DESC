"""Classes for parameterized 3D umbilic space curves."""

import os
import pdb
import numpy as np

from desc.backend import jnp, put
from desc.basis import FourierSeries
from desc.grid import LinearGrid
from desc.optimizable import optimizable_parameter
from desc.transform import Transform
from desc.utils import check_posint, copy_coeffs

from .core import UmbilicCurve

__all__ = ["FourierUmbilicCurve"]


class FourierUmbilicCurve(UmbilicCurve):
    """Curve parameterized by Fourier series for A in terms of toroidal angle phi.

    Parameters
    ----------
    A_n: array-like
        Fourier coefficients fo Z.
    modes_A : array-like, optional
        Mode numbers associated with Z_n, If not given defaults to [-n:n]].
    NFP : int
        Number of field periods.
    NFP_umbilic_factor : float
        Rational number of the form 1/integer with integer>=1.
        This is needed for the umbilic torus design.
    sym : bool
        Whether to enforce stellarator symmetry.
    name : str
        Name for this curve.

    """

    _io_attrs_ = UmbilicCurve._io_attrs_ + [
        "_A_n",
        "_A_basis",
        "_sym",
        "_NFP",
        "_NFP_umbilic_factor",
    ]

    def __init__(
        self,
        A_n=0,
        modes_A=None,
        NFP=1,
        NFP_umbilic_factor=1,
        sym="auto",
        name="",
    ):
        super().__init__(name)
        A_n = np.atleast_1d(A_n)
        if modes_A is None:
            modes_A = np.arange(-(A_n.size // 2), A_n.size // 2 + 1)

        if A_n.size == 0:
            A_n = np.array([0.0])
            modes_A = np.array([0])

        modes_A = np.asarray(modes_A)

        assert A_n.size == modes_A.size, "A_n size and modes_A must be the same size"

        assert issubclass(modes_A.dtype.type, np.integer)

        if sym == "auto":
            if np.all(A_n[modes_A >= 0] == 0):
                sym = True
            else:
                sym = False
        self._sym = sym
        NA = np.max(abs(modes_A))
        N = NA
        self._NFP = check_posint(NFP, "NFP", False)
        self._NFP_umbilic_factor = check_posint(
            NFP_umbilic_factor, "NFP_umbilic_factor", False
        )
        self._A_basis = FourierSeries(
            N,
            int(NFP),
            NFP_umbilic_factor=int(NFP_umbilic_factor),
            sym="sin" if sym else False,
        )

        self._A_n = copy_coeffs(A_n, modes_A, self.A_basis.modes[:, 2])

    @property
    def sym(self):
        """bool: Whether or not the curve is stellarator symmetric."""
        return self._sym

    @property
    def A_basis(self):
        """Spectral basis for A_Fourier series."""
        return self._A_basis

    @property
    def NFP(self):
        """Number of field periods."""
        return self._NFP

    @property
    def NFP_umbilic_factor(self):
        """NFP umbilic factor. Effective NFP -> NFP/NFP_umbilic_factor."""
        return self.__dict__.setdefault("_NFP_umbilic_factor", 1)

    def _NFP_umbilic_factor(self):
        """NFP umbilic factor. Effective NFP -> NFP/NFP_umbilic_factor."""
        self._NFP_umbilic_factor = self.NFP_umbilic_factor

    @property
    def N(self):
        """Maximum mode number."""
        return self.A_basis.N

    def get_coeffs(self, n):
        """Get Fourier coefficients for given mode number(s)."""
        ## CURRENTLY ONLY OUTPUTS COEFFICIENTS FOR NEGATIVE n
        ## values
        n = np.atleast_1d(n).astype(int)
        A = np.zeros_like(n).astype(float)

        idxA = np.where(n[:, np.newaxis] == self.A_basis.modes[:, 2])

        A[idxA[0]] = self.A_n[idxA[1]]
        return A

    def set_coeffs(self, n, A=None):
        """Set specific Fourier coefficients."""
        n, A = np.atleast_1d(n), np.atleast_1d(A)
        A = np.broadcast_to(A, n.shape)
        for nn, AA in zip(n, A):
            if AA is not None:
                idxA = self.A_basis.get_idx(0, 0, nn)
                self.A_n = put(self.A_n, idxA, AA)

    @optimizable_parameter
    @property
    def A_n(self):
        """Spectral coefficients for Z."""
        return self._A_n

    @A_n.setter
    def A_n(self, new):
        if len(new) == self.A_basis.num_modes:
            self._A_n = jnp.asarray(new)
        else:
            raise ValueError(
                f"A_n should have the same size as the basis, got {len(new)} for "
                + f"basis with {self.A_basis.num_modes} modes"
            )

    @classmethod
    def from_values(cls, coords, N=10, NFP=1, NFP_umbilic_factor=1, name="", sym=False):
        """Fit coordinates to FourierRZCurve representation.

        Parameters
        ----------
        coords: ndarray, shape (num_coords,2)
            coordinates theta, zeta, the different of which is fit with a FourierSeries
        N : int
            Fourier resolution of the new R,Z representation.
        NFP : int
            Number of field periods, the curve will have a discrete toroidal symmetry
            according to NFP.
        NFP_umbilic_factor : int
            Umbilic factor to fit curves that go around multiple times toroidally before
            closing on themselves.
        sym : bool
            Whether to enforce stellarator symmetry.
        basis : {"rpz", "xyz"}
            basis for input coordinates. Defaults to "rpz"
        name : str
            Name for this curve.

        Returns
        -------
        curve : FourierZSeries
            New representation of the curve parameterized by Fourier series for Z.

        """
        phi = coords[:, 0]
        A = coords[:, 1]

        grid = LinearGrid(zeta=phi, NFP=1, NFP_umbilic_factor=1, sym=sym)
        basis = FourierSeries(N=N, NFP=1, sym=sym)
        transform = Transform(grid, basis, build_pinv=True)
        A_n = transform.fit(A)

        #pdb.set_trace()
        return FourierUmbilicCurve(
            A_n=A_n,
            NFP=NFP,
            NFP_umbilic_factor=NFP_umbilic_factor,
            modes_A=basis.modes[:, 2],
            sym=sym,
            name=name,
        )
