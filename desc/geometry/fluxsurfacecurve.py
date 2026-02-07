"""Classes for parameterized 3D space curves constrained to lie in flux surfaces."""

import numpy as np

from desc.backend import jnp, put
from desc.basis import FourierSeries
from desc.grid import LinearGrid
from desc.optimizable import optimizable_parameter
from desc.transform import Transform
from desc.utils import check_posint, copy_coeffs

from .core import FluxSurfaceCurve

__all__ = ["FourierUmbilicCurve"]


class FourierUmbilicCurve(FluxSurfaceCurve):
    r"""Curve parameterized by Fourier series in terms of toroidal angle phi.

    Specific parameterization introduced for study of umbilic curves in [1].
    Given in DESC coordinates by
    \theta = (m_umbilic/n_umbilic*NFP)\\phi
                    + (1/n_umbilic)\\sum_{n=0}^{N} a_n cos( (n*NFP/n_umbilic) phi)
                    + (1/n_umbilic)\\sum_{n=-N}^{-1} a_n sin( (|n|*NFP/n_umbilic) phi)

    References
    ----------
    [1] https://arxiv.org/abs/2505.04211.
    Omnigenous Umbilic Stellarators.
    R. Gaur, D. Panici, T.M. Elder, M. Landreman, K.E. Unalmis,
    Y. Elmacioglu. D. Dudt, R. Conlin, E. Kolemen.


    Parameters
    ----------
    a_n: array-like
        Fourier coefficients of curve's angular displacement.
    modes_UC : array-like, optional
        Integer mode numbers associated with a_n. If not given,
        defaults to [-N:N] where N = len(a_n)//2.
    NFP : int
        Number of field periods.
    n_umbilic : int
        Prefactor of the form 1/n_umbilic modifying NFP.
        Curve closes after n_umbilic/gcd(n_umbilic, NFP) transits.
        Default is n_umbilic = 1.
    m_umbilic : int
        Parameter arising from umbilic torus parameterization, determining
        the average slope of the curve in the (theta,phi) plane.
        Should satisfy gcd(n_umbilic, m_umbilic)=1.
        Default is m_umbilic = 1.
    sym : bool
        Whether to enforce stellarator symmetry.
    name : str
        Name for this curve.
    """

    _io_attrs_ = FluxSurfaceCurve._io_attrs_ + [
        "_a_n",
        "_UC_basis",
        "_sym",
        "_NFP",
        "_n_umbilic",
        "_m_umbilic",
    ]

    _static_attrs = FluxSurfaceCurve._static_attrs + [
        "_UC_basis",
        "_sym",
        "_NFP",
        "_n_umbilic",
        "_m_umbilic",
    ]

    def __init__(
        self,
        a_n=None,
        modes_UC=None,
        NFP=1,
        n_umbilic=1,
        m_umbilic=1,
        sym="auto",
        name="",
    ):
        super().__init__(name)

        if a_n is None:
            a_n = np.array([0.0, 0.0, 0.0])
            modes_UC = np.array([-1, 0, 1])
        a_n = np.atleast_1d(a_n)

        if modes_UC is None:
            modes_UC = np.arange(-(a_n.size // 2), a_n.size // 2 + 1)

        modes_UC = np.asarray(modes_UC)

        assert a_n.size == modes_UC.size, "a_n and modes_UC must be the same size"
        assert (
            jnp.gcd(n_umbilic, m_umbilic) == 1
        ), "n_umbilic and m_umbilic should have gcd = 1"
        assert issubclass(modes_UC.dtype.type, np.integer)

        if sym == "auto":
            if np.all(a_n[modes_UC >= 0] == 0):
                sym = True
            else:
                sym = False
        self._sym = sym
        N = np.max(abs(modes_UC))
        self._NFP = check_posint(NFP, "NFP", False)
        self._n_umbilic = check_posint(n_umbilic, "n_umbilic", False)
        self._m_umbilic = check_posint(m_umbilic, "m_umbilic", False)
        self._N_scaling = self._n_umbilic
        self._UC_basis = FourierSeries(
            N,
            int(NFP),
            N_scaling=int(n_umbilic),
            sym="sin" if sym else False,
        )
        self._a_n = copy_coeffs(a_n, modes_UC, self.UC_basis.modes[:, 2])

    def _set_up(self):
        super()._set_up()
        self._a_n = jnp.atleast_1d(self._a_n)
        self._n_umbilic = int(self._n_umbilic)
        self._m_umbilic = int(self._m_umbilic)
        self._UC_basis._N_scaling = self._n_umbilic

    def compute(
        self,
        names,
        grid=None,
        params=None,
        transforms=None,
        data=None,
        override_grid=True,
        **kwargs,
    ):
        """Compute the quantity given by name on grid.

        Parameters
        ----------
        names : str or array-like of str
            Name(s) of the quantity(s) to compute.
        grid : Grid or int, optional
            Grid of coordinates to evaluate at. Defaults to a Linear grid.
            If an integer, uses that many equally spaced points.
        params : dict of ndarray
            Parameters from the equilibrium. Defaults to attributes of self.
        transforms : dict of Transform
            Transforms for R, Z, lambda, etc. Default is to build from grid
        data : dict of ndarray
            Data computed so far, generally output from other compute functions.
            Any vector v = v¹ R̂ + v² ϕ̂ + v³ Ẑ should be given in components
            v = [v¹, v², v³] where R̂, ϕ̂, Ẑ are the normalized basis vectors
            of the cylindrical coordinates R, ϕ, Z.
        override_grid : bool
            If True, override the user supplied grid if necessary and use a full
            resolution grid to compute quantities and then downsample to user requested
            grid. If False, uses only the user specified grid, which may lead to
            inaccurate values for surface or volume averages.

        Returns
        -------
        data : dict of ndarray
            Computed quantity and intermediate variables.

        """
        kwargs["n_umbilic"] = self._n_umbilic
        kwargs["m_umbilic"] = self._m_umbilic
        kwargs["NFP"] = self._NFP

        data = super().compute(
            names, grid, params, transforms, data, override_grid, **kwargs
        )
        return data

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
    def n_umbilic(self):
        """NFP umbilic factor. Effective NFP -> NFP/n_umbilic."""
        return self.__dict__.setdefault("_n_umbilic", 1)

    @property
    def N_scaling(self):
        """Alias for n_umbilic."""
        return self.__dict__.setdefault("_n_umbilic", 1)

    @property
    def m_umbilic(self):
        """Slope parameter for curve in (theta,phi) plane."""
        return self.__dict__.setdefault("_m_umbilic", 1)

    @property
    def N(self):
        """Maximum mode number."""
        return self.UC_basis.N

    def change_resolution(
        self, N=None, NFP=None, n_umbilic=None, m_umbilic=None, sym=None
    ):
        """Change the maximum toroidal resolution."""
        N = check_posint(N, "N")
        NFP = check_posint(NFP, "NFP")
        n_umbilic = check_posint(n_umbilic, "n_umbilic")
        m_umbilic = check_posint(m_umbilic, "m_umbilic")
        if (
            ((N is not None) and (N != self.N))
            or ((NFP is not None) and (NFP != self.NFP))
            or ((n_umbilic is not None) and (n_umbilic != self.n_umbilic))
            or ((m_umbilic is not None) and (m_umbilic != self.m_umbilic))
            or ((sym is not None) and (sym != self.sym))
        ):
            self._NFP = int(NFP if NFP is not None else self.NFP)
            self._n_umbilic = int(
                n_umbilic if n_umbilic is not None else self.n_umbilic
            )
            self._m_umbilic = int(
                m_umbilic if m_umbilic is not None else self.m_umbilic
            )
            assert (
                jnp.gcd(self.n_umbilic, self.m_umbilic) == 1
            ), "n_umbilic and m_umbilic should have gcd = 1"

            self._sym = bool(sym) if sym is not None else self.sym
            N = int(N if N is not None else self.N)

            UC_modes_old = self.UC_basis.modes
            self.UC_basis.change_resolution(
                N=N,
                NFP=self.NFP,
                N_scaling=self.n_umbilic,
                sym="cos" if self.sym else self.sym,
            )
            self.a_n = copy_coeffs(self.a_n, UC_modes_old, self.UC_basis.modes)

    def get_coeffs(self, n):
        """Get Fourier coefficients for given mode number(s)."""
        n = np.atleast_1d(n).astype(int)
        a_n = np.zeros_like(n).astype(float)

        idxUC = np.where(n[:, np.newaxis] == self.UC_basis.modes[:, 2])

        a_n[idxUC[0]] = self._a_n[idxUC[1]]
        return a_n

    def set_coeffs(self, n, UC=None):
        """Set specific Fourier coefficients."""
        n, UC = np.atleast_1d(n), np.atleast_1d(UC)
        UC = np.broadcast_to(UC, n.shape)
        for nn, nUC in zip(n, UC):
            if nUC is not None:
                idxUC = self.UC_basis.get_idx(0, 0, nn)
                self.a_n = put(self.a_n, idxUC, nUC)

    @optimizable_parameter
    @property
    def a_n(self):
        """Spectral coefficients for a_n, angular displacement along the curve ."""
        return self._a_n

    @a_n.setter
    def a_n(self, new):
        if len(new) == self.UC_basis.num_modes:
            self._a_n = jnp.asarray(new)
        else:
            raise ValueError(
                f"a_n should have the same size as the basis, got {len(new)} for "
                + f"basis with {self.UC_basis.num_modes} modes"
            )

    @classmethod
    def from_values(
        cls, coords, N=10, NFP=1, n_umbilic=1, m_umbilic=1, name="", sym=False
    ):
        """Fit a FourierUmbilicCurve to given (theta,phi) values.

        Parameters
        ----------
        coords: ndarray, shape (num_coords,2)
            Coordinates theta, phi along the curve.
        N : int
            Fourier resolution of the curve parameterization. Resulting curve will have
            modes UC_modes = [-N:N] populated.
        NFP : int
            Number of field periods, the curve will have a discrete toroidal symmetry
            according to NFP.
        n_umbilic : int
            Umbilic factor to fit curves that go around multiple times toroidally before
            closing on themselves.
        m_umbilic : int
            Umbilic factor to set the average slope of curve in the (theta,phi) plane.
        sym : bool
            Whether to enforce stellarator symmetry.
        name : str
            Name for this curve.

        Returns
        -------
        curve : FourierUmbilicCurve
            New representation of the curve parameterized by Fourier series for phi.

        """
        theta = coords[:, 0]
        phi = coords[:, 1]
        UC = n_umbilic * theta - m_umbilic * NFP * phi

        # Will fit a basis with period n_umbilic/NFP*2pi,
        # so mod input phi values by this
        period = n_umbilic / NFP * 2 * np.pi
        phi = phi % period

        # Sort and remove duplicate values of phi
        phi, idx = np.unique(phi, return_index=True)
        UC = UC[idx]

        grid = LinearGrid(zeta=phi, NFP=NFP, N_scaling=n_umbilic, sym=sym)
        basis = FourierSeries(N=N, NFP=NFP, N_scaling=n_umbilic, sym=sym)
        transform = Transform(grid, basis, build_pinv=True)
        a_n = transform.fit(UC)

        return FourierUmbilicCurve(
            a_n=a_n,
            NFP=NFP,
            n_umbilic=n_umbilic,
            m_umbilic=m_umbilic,
            modes_UC=basis.modes[:, 2],
            sym=sym,
            name=name,
        )
