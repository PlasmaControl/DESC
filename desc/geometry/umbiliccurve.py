"""Classes for parameterized 3D umbilic space curves."""

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
    """Curve parameterized by Fourier series for UC in terms of toroidal angle phi.

    Parameters
    ----------
    UC_n: array-like
        Fourier coefficients fo Z.
    modes_UC : array-like, optional
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
        "_UC_n",
        "_UC_basis",
        "_sym",
        "_NFP",
        "_NFP_umbilic_factor",
    ]

    def __init__(
        self,
        UC_n=0,
        modes_UC=None,
        NFP=1,
        NFP_umbilic_factor=1,
        sym="auto",
        name="",
    ):
        super().__init__(name)
        UC_n = np.atleast_1d(UC_n)
        if modes_UC is None:
            modes_UC = np.arange(-(UC_n.size // 2), UC_n.size // 2 + 1)

        if UC_n.size == 0:
            UC_n = np.array([0.0])
            modes_UC = np.array([0])

        modes_UC = np.asarray(modes_UC)

        assert (
            UC_n.size == modes_UC.size
        ), "UC_n size and modes_UC must be the same size"

        assert issubclass(modes_UC.dtype.type, np.integer)

        if sym == "auto":
            if np.all(UC_n[modes_UC >= 0] == 0):
                sym = True
            else:
                sym = False
        self._sym = sym
        NUC = np.max(abs(modes_UC))
        N = NUC
        self._NFP = check_posint(NFP, "NFP", False)
        self._NFP_umbilic_factor = check_posint(
            NFP_umbilic_factor, "NFP_umbilic_factor", False
        )
        self._UC_basis = FourierSeries(
            N,
            int(NFP),
            NFP_umbilic_factor=int(NFP_umbilic_factor),
            sym="sin" if sym else False,
        )

        self._UC_n = copy_coeffs(UC_n, modes_UC, self.UC_basis.modes[:, 2])

    @property
    def sym(self):
        """bool: Whether or not the curve is stellarator symmetric."""
        return self._sym

    @property
    def UC_basis(self):
        """Spectral basis for UC_Fourier series."""
        return self._UC_basis

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
        return self.UC_basis.N

    def get_coeffs(self, n):
        """Get Fourier coefficients for given mode number(s)."""
        ## CURRENTLY ONLY OUTPUTS COEFFICIENTS FOR NEGATIVE n
        ## values
        n = np.atleast_1d(n).astype(int)
        UC = np.zeros_like(n).astype(float)

        idxUC = np.where(n[:, np.newaxis] == self.UC_basis.modes[:, 2])

        UC[idxUC[0]] = self.UC_n[idxUC[1]]
        return UC

    def set_coeffs(self, n, UC=None):
        """Set specific Fourier coefficients."""
        n, UC = np.atleast_1d(n), np.atleast_1d(UC)
        UC = np.broadcast_to(UC, n.shape)
        for nn, nUC in zip(n, UC):
            if nUC is not None:
                idxUC = self.UC_basis.get_idx(0, 0, nn)
                self.UC_n = put(self.UC_n, idxUC, nUC)

    @optimizable_parameter
    @property
    def UC_n(self):
        """Spectral coefficients for Z."""
        return self._UC_n

    @UC_n.setter
    def UC_n(self, new):
        if len(new) == self.UC_basis.num_modes:
            self._UC_n = jnp.asarray(new)
        else:
            raise ValueError(
                f"UC_n should have the same size as the basis, got {len(new)} for "
                + f"basis with {self.UC_basis.num_modes} modes"
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
        name : str
            Name for this curve.

        Returns
        -------
        curve : FourierZSeries
            New representation of the curve parameterized by Fourier series for Z.

        """
        phi = coords[:, 0]
        UC = coords[:, 1]

        grid = LinearGrid(zeta=phi, NFP=1, NFP_umbilic_factor=1, sym=sym)
        basis = FourierSeries(N=N, NFP=1, sym=sym)
        transform = Transform(grid, basis, build_pinv=True)
        UC_n = transform.fit(UC)

        return FourierUmbilicCurve(
            UC_n=UC_n,
            NFP=NFP,
            NFP_umbilic_factor=NFP_umbilic_factor,
            modes_UC=basis.modes[:, 2],
            sym=sym,
            name=name,
        )
