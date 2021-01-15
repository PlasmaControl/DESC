import math
import numpy as np
from netCDF4 import Dataset, stringtochar

from desc.backend import put
from desc.utils import Tristate, sign
from desc.grid import LinearGrid
from desc.basis import DoubleFourierSeries, FourierZernikeBasis, jacobi
from desc.transform import Transform
from desc.configuration import Configuration
from desc.boundary_conditions import BoundaryConstraint


class VMECIO:
    """Performs input from VMEC netCDF files to DESC Configurations and vice-versa."""

    @classmethod
    def load(
        cls, path: str, L: int = -1, M: int = -1, N: int = -1, index: str = "ansi"
    ) -> Configuration:
        """Loads a VMEC netCDF file as a Configuration.

        Parameters
        ----------
        path : str
            File path of input data.
        L : int, optional
            Radial resolution. Default determined by index.
        M : int, optional
            Poloidal resolution. Default = MPOL-1 from VMEC solution.
        N : int, optional
            Toroidal resolution. Default = NTOR from VMEC solution.
        index : str, optional
            Type of Zernike indexing scheme to use. (Default = 'ansi')

        Returns
        -------
        eq: Configuration
            Configuration that resembles the VMEC data.

        """
        file = Dataset(path, mode="r")
        inputs = {}

        # parameters
        inputs["Psi"] = file.variables["phi"][-1]
        inputs["NFP"] = int(file.variables["nfp"][0])
        inputs["M"] = M if M > 0 else int(file.variables["mpol"][0] - 1)
        inputs["N"] = N if N >= 0 else int(file.variables["ntor"][0])
        inputs["index"] = index
        default_L = {
            "ansi": inputs["M"],
            "fringe": 2 * inputs["M"],
            "chevron": inputs["M"],
            "house": 2 * inputs["M"],
        }
        inputs["L"] = L if L >= 0 else default_L[inputs["index"]]

        # data
        xm = file.variables["xm"][:].filled()
        xn = file.variables["xn"][:].filled() / inputs["NFP"]
        rmnc = file.variables["rmnc"][:].filled()
        zmns = file.variables["zmns"][:].filled()
        lmns = file.variables["lmns"][:].filled()
        try:
            rmns = file.variables["rmns"][:].filled()
            zmnc = file.variables["zmnc"][:].filled()
            lmnc = file.variables["lmnc"][:].filled()
            inputs["sym"] = False
        except:
            rmns = np.zeros_like(rmnc)
            zmnc = np.zeros_like(zmns)
            lmnc = np.zeros_like(lmns)
            inputs["sym"] = True

        # profiles
        preset = file.dimensions["preset"].size
        p0 = file.variables["presf"][0] / file.variables["am"][0]
        inputs["profiles"] = np.zeros((preset, 3))
        inputs["profiles"][:, 0] = np.arange(0, 2 * preset, 2)
        inputs["profiles"][:, 1] = file.variables["am"][:] * p0
        inputs["profiles"][:, 2] = file.variables["ai"][:]

        file.close

        # boundary
        m, n, Rb_mn = cls._ptolemy_identity_fwd(xm, xn, s=rmns[-1, :], c=rmnc[-1, :])
        m, n, Zb_mn = cls._ptolemy_identity_fwd(xm, xn, s=zmns[-1, :], c=zmnc[-1, :])
        inputs["boundary"] = np.vstack((m, n, Rb_mn, Zb_mn)).T

        # initialize Configuration
        eq = Configuration(inputs=inputs)

        # R
        m, n, R_mn = cls._ptolemy_identity_fwd(xm, xn, s=rmns, c=rmnc)
        eq.R_lmn = cls._fourier_to_zernike(m, n, R_mn, eq.R_basis)

        # Z
        m, n, Z_mn = cls._ptolemy_identity_fwd(xm, xn, s=zmns, c=zmnc)
        eq.Z_lmn = cls._fourier_to_zernike(m, n, Z_mn, eq.Z_basis)

        # lambda
        m, n, L_mn = cls._ptolemy_identity_fwd(xm, xn, s=lmns, c=lmnc)
        eq.L_lmn = cls._fourier_to_zernike(m, n, L_mn, eq.L_basis)

        # apply boundary conditions
        #   BC = BoundaryConstraint(eq.R_basis, eq.Z_basis, eq.L_basis)
        #   eq.x = BC.make_feasible(eq.x)

        return eq

    @classmethod
    def save(cls, eq: Configuration, path: str, surfs: int = 128) -> None:
        """Saves a Configuration as a netCDF file in the VMEC format.

        Parameters
        ----------
        eq : Configuration
            Configuration to save.
        path : str
            File path of output data.
        surfs: int (Default = 128)
            Number of flux surfaces to interpolate at.

        Returns
        -------
        None

        """
        file = Dataset(path, mode="w")

        """ VMEC netCDF file is generated in VMEC2000/Sources/Input_Output/wrout.f
            see lines 300+ for full list of included variables
        """

        Psi = eq.Psi
        NFP = eq.NFP
        M = eq.M
        N = eq.N

        s_full = np.linspace(0, 1, surfs)
        s_half = s_full[0:-1] + 0.5 / (surfs - 1)
        r_full = np.sqrt(s_full)
        r_half = np.sqrt(s_half)
        full_grid = LinearGrid(rho=r_full)
        half_grid = LinearGrid(rho=r_half)

        p_transform_full = Transform(full_grid, eq.p_basis)
        p_transform_half = Transform(half_grid, eq.p_basis)
        i_transform_full = Transform(full_grid, eq.i_basis)
        i_transform_half = Transform(half_grid, eq.i_basis)

        # dimensions
        file.createDimension("radius", surfs)  # number of flux surfaces
        file.createDimension(
            "mn_mode", (2 * N + 1) * M + N + 1
        )  # number of Fourier modes
        file.createDimension("mn_mode_nyq", None)  # used for Nyquist quantities
        file.createDimension("n_tor", 1)  # number of axis guess Fourier modes
        file.createDimension("preset", 21)  # max dimension of profile inputs
        file.createDimension("ndfmax", 101)  # used for am_aux & ai_aux
        file.createDimension("time", 100)  # used for fsqrt & wdot
        file.createDimension("dim_00001", 1)
        file.createDimension("dim_00020", 20)
        file.createDimension("dim_00100", 100)
        file.createDimension("dim_00200", 200)
        preset = file.dimensions["preset"].size

        # variables

        lfreeb = file.createVariable("lfreeb", np.int32, ("dim_00001",))
        lfreeb.long_name = "free boundary logical (0 = fixed boundary)"
        lfreeb[:] = 0

        lasym = file.createVariable("lasym__logical__", np.int32, ("dim_00001",))
        lasym.long_name = "asymmetry logical (0 = stellarator symmetry)"
        lasym[:] = int(not eq.sym)

        nfp = file.createVariable("nfp", np.int32, ("dim_00001",))
        nfp.long_name = "number of field periods"
        nfp[:] = NFP

        ns = file.createVariable("ns", np.int32, ("dim_00001",))
        ns.long_name = "number of flux surfaces"
        ns[:] = surfs

        mpol = file.createVariable("mpol", np.int32, ("dim_00001",))
        mpol.long_name = "number of poloidal Fourier modes"
        mpol[:] = M + 1

        ntor = file.createVariable("ntor", np.int32, ("dim_00001",))
        ntor.long_name = "number of positive toroidal Fourier modes"
        ntor[:] = N

        mnmax = file.createVariable("mnmax", np.int32, ("dim_00001",))
        mnmax.long_name = "total number of Fourier modes"
        mnmax[:] = file.dimensions["mn_mode"].size

        xm = file.createVariable("xm", np.float64, ("mn_mode",))
        xm.long_name = "poloidal mode numbers"
        xm[:] = np.tile(np.linspace(0, M, M + 1), (2 * N + 1, 1)).T.flatten()[
            -file.dimensions["mn_mode"].size :
        ]

        xn = file.createVariable("xn", np.float64, ("mn_mode",))
        xn.long_name = "toroidal mode numbers"
        xn[:] = np.tile(np.linspace(-N, N, 2 * N + 1) * NFP, M + 1)[
            -file.dimensions["mn_mode"].size :
        ]

        gamma = file.createVariable("gamma", np.float64, ("dim_00001",))
        gamma.long_name = "compressibility index (0 = pressure prescribed)"
        gamma[:] = 0

        signgs = file.createVariable("signgs", np.float64, ("dim_00001",))
        signgs.long_name = "sign of coordinate system jacobian"
        signgs[:] = 1  # TODO: don't hard-code this

        am = file.createVariable("am", np.float64, ("preset",))
        am.long_name = "pressure coefficients"
        am.units = "Pa"
        am[:] = np.zeros((file.dimensions["preset"].size,))
        dim = min(preset, eq.p_l.size)
        am[0:dim] = eq.p_l[0:dim]

        ai = file.createVariable("ai", np.float64, ("preset",))
        ai.long_name = "rotational transform coefficients"
        ai[:] = np.zeros((file.dimensions["preset"].size,))
        dim = min(preset, eq.p_l.size)
        ai[0:dim] = eq.i_l[0:dim]

        ac = file.createVariable("ac", np.float64, ("preset",))
        ac.long_name = "normalized toroidal current density coefficients"
        ac[:] = np.zeros((file.dimensions["preset"].size,))

        power_series = stringtochar(
            np.array(
                ["power_series         "], "S" + str(file.dimensions["preset"].size)
            )
        )

        pmass_type = file.createVariable("pmass_type", "S1", ("preset",))
        pmass_type.long_name = "parameterization of pressure function"
        pmass_type[:] = power_series

        piota_type = file.createVariable("piota_type", "S1", ("preset",))
        piota_type.long_name = "parameterization of rotational transform function"
        piota_type[:] = power_series

        pcurr_type = file.createVariable("pcurr_type", "S1", ("preset",))
        pcurr_type.long_name = "parameterization of current density function"
        pcurr_type[:] = power_series

        presf = file.createVariable("presf", np.float64, ("radius",))
        presf.long_name = "pressure on full mesh"
        presf.units = "Pa"
        presf[:] = p_transform_full.transform(eq.p_l)

        pres = file.createVariable("pres", np.float64, ("radius",))
        pres.long_name = "pressure on half mesh"
        pres.units = "Pa"
        pres[0] = 0
        pres[1:] = p_transform_half.transform(eq.p_l)

        mass = file.createVariable("mass", np.float64, ("radius",))
        mass.long_name = "mass on half mesh"
        mass.units = "Pa"
        mass[:] = pres[:]

        iotaf = file.createVariable("iotaf", np.float64, ("radius",))
        iotaf.long_name = "rotational transform on full mesh"
        iotaf[:] = i_transform_full.transform(eq.i_l)

        iotas = file.createVariable("iotas", np.float64, ("radius",))
        iotas.long_name = "rotational transform on half mesh"
        iotas[0] = 0
        iotas[1:] = i_transform_half.transform(eq.i_l)

        phi = file.createVariable("phi", np.float64, ("radius",))
        phi.long_name = "toroidal flux"
        phi.units = "Wb"
        phi[:] = np.linspace(0, Psi, surfs)

        phipf = file.createVariable("phipf", np.float64, ("radius",))
        phipf.long_name = "d(phi)/ds: toroidal flux derivative"
        phipf[:] = Psi * np.ones((surfs,))

        phips = file.createVariable("phips", np.float64, ("radius",))
        phips.long_name = (
            "d(phi)/ds * sign(g)/2pi: toroidal flux derivative on half mesh"
        )
        phips[0] = 0
        phips[1:] = phipf[1:] * signgs[:] / (2 * np.pi)

        chi = file.createVariable("chi", np.float64, ("radius",))
        chi.long_name = "poloidal flux"
        chi.units = "Wb"
        chi[:] = phi[:] * signgs[:]

        chipf = file.createVariable("chipf", np.float64, ("radius",))
        chipf.long_name = "d(chi)/ds: poloidal flux derivative"
        chipf[:] = phipf[:] * iotaf[:]

        # indepentent variables (exact conversion)

        # R
        rmnc = file.createVariable("rmnc", np.float64, ("radius", "mn_mode"))
        rmnc.long_name = "cos(m*t-n*p) component of cylindrical R, on full mesh"
        rmnc.units = "m"
        if not eq.sym:
            rmns = file.createVariable("rmns", np.float64, ("radius", "mn_mode"))
            rmns.long_name = "sin(m*t-n*p) component of cylindrical R, on full mesh"
            rmns.units = "m"
        m, n, x_mn = cls._zernike_to_fourier(eq.R_lmn, basis=eq.R_basis, rho=r_full)
        xm, xn, s, c = cls._ptolemy_identity_rev(m, n, x_mn)
        rmnc[:] = c
        if not eq.sym:
            rmns[:] = s

        # Z
        zmns = file.createVariable("zmns", np.float64, ("radius", "mn_mode"))
        zmns.long_name = "sin(m*t-n*p) component of cylindrical Z, on full mesh"
        zmns.units = "m"
        if not eq.sym:
            zmnc = file.createVariable("zmnc", np.float64, ("radius", "mn_mode"))
            zmnc.long_name = "cos(m*t-n*p) component of cylindrical Z, on full mesh"
            zmnc.units = "m"
        m, n, x_mn = cls._zernike_to_fourier(eq.Z_lmn, basis=eq.Z_basis, rho=r_full)
        xm, xn, s, c = cls._ptolemy_identity_rev(m, n, x_mn)
        zmns[:] = s
        if not eq.sym:
            zmnc[:] = c

        # lambda
        lmns = file.createVariable("lmns", np.float64, ("radius", "mn_mode"))
        lmns.long_name = "sin(m*t-n*p) component of lambda, on half mesh"
        lmns.units = "rad"
        if not eq.sym:
            lmnc = file.createVariable("lmnc", np.float64, ("radius", "mn_mode"))
            lmnc.long_name = "cos(m*t-n*p) component of lambda, on half mesh"
            lmnc.units = "rad"
        m, n, x_mn = cls._zernike_to_fourier(eq.L_lmn, basis=eq.L_basis, rho=r_half)
        xm, xn, s, c = cls._ptolemy_identity_rev(m, n, x_mn)
        lmns[1:, :] = s
        if not eq.sym:
            lmnc[1:, :] = c

        # derived quantities (approximate conversion)

        MM = 2 * math.ceil(1.5 * M) + 1
        NN = 2 * math.ceil(1.5 * N) + 1
        grid = LinearGrid(M=MM, N=NN, NFP=NFP)

        if eq.sym:
            sin_basis = DoubleFourierSeries(M=M, N=N, NFP=NFP, sym=Tristate(False))
            cos_basis = DoubleFourierSeries(M=M, N=N, NFP=NFP, sym=Tristate(True))
            sin_transform = Transform(grid=grid, basis=sin_basis)
            cos_transform = Transform(grid=grid, basis=cos_basis)
        else:
            full_basis = DoubleFourierSeries(M=M, N=N, NFP=NFP, sym=Tristate(None))
            full_transform = Transform(grid=grid, basis=full_basis)

        # g
        gmnc = file.createVariable("gmnc", np.float64, ("radius", "mn_mode_nyq"))
        gmnc.long_name = "cos(m*t-n*p) component of jacobian, on half mesh"
        gmnc.units = "m"
        m = cos_basis.modes[:, 1]
        n = cos_basis.modes[:, 2]
        if not eq.sym:
            gmns = file.createVariable("gmns", np.float64, ("radius", "mn_mode_nyq"))
            gmns.long_name = "sin(m*t-n*p) component of jacobian, on half mesh"
            gmns.units = "m"
            m = full_basis.modes[:, 1]
            n = full_basis.modes[:, 2]
        x_mn = np.zeros((surfs - 1, m.size))
        for k in range(surfs - 1):
            grid = LinearGrid(M=MM, N=NN, NFP=NFP, rho=r_half[k])
            data = eq.compute_jacobian(grid)["g"]
            if eq.sym:
                x_mn[k, :] = cos_transform.fit(data)
            else:
                x_mn[k, :] = full_transform.fit(data)
        xm, xn, s, c = cls._ptolemy_identity_rev(m, n, x_mn)
        gmnc[1:, :] = c
        if not eq.sym:
            gmns[1:, :] = s

        # |B|
        bmnc = file.createVariable("bmnc", np.float64, ("radius", "mn_mode_nyq"))
        bmnc.long_name = "cos(m*t-n*p) component of |B|, on half mesh"
        bmnc.units = "m"
        m = cos_basis.modes[:, 1]
        n = cos_basis.modes[:, 2]
        if not eq.sym:
            bmns = file.createVariable("bmns", np.float64, ("radius", "mn_mode_nyq"))
            bmns.long_name = "sin(m*t-n*p) component of |B|, on half mesh"
            bmns.units = "m"
            m = full_basis.modes[:, 1]
            n = full_basis.modes[:, 2]
        x_mn = np.zeros((surfs - 1, m.size))
        for k in range(surfs - 1):
            grid = LinearGrid(M=MM, N=NN, NFP=NFP, rho=r_half[k])
            data = eq.compute_magnetic_field(grid)["|B|"]
            if eq.sym:
                x_mn[k, :] = cos_transform.fit(data)
            else:
                x_mn[k, :] = full_transform.fit(data)
        xm, xn, s, c = cls._ptolemy_identity_rev(m, n, x_mn)
        bmnc[1:, :] = c
        if not eq.sym:
            bmns[1:, :] = s

        file.close

    @staticmethod
    def _ptolemy_identity_fwd(m_0, n_0, s, c):
        """Converts from double-angle form:
            s*sin(m*theta-n*phi) + c*cos(m*theta-n*phi)
        to a double Fourier series of the form:
            ss*sin(m*theta)*sin(n*phi) + sc*sin(m*theta)*cos(n*phi) +
            cs*cos(m*theta)*sin(n*phi) + cc*cos(m*theta)*cos(n*phi)
        using Ptolemy's sum and difference formulas.

        Parameters
        ----------
        m_0 : ndarray
            Poloidal mode numbers of the double-angle Fourier basis.
        n_0 : ndarray
            Toroidal mode numbers of the double-angle Fourier basis.
        s : ndarray, shape(surfs,num_modes), optional
            Coefficients of sin(m*theta-n*phi) terms.
            Each row is a separate flux surface.
        c : ndarray, shape(surfs,num_modes), optional
            Coefficients of cos(m*theta-n*phi) terms.
            Each row is a separate flux surface.

        Returns
        -------
        m_1 : ndarray, shape(num_modes,)
            Poloidal mode numbers of the double Fourier basis.
        n_1 : ndarray, shape(num_modes,)
            Toroidal mode numbers of the double Fourier basis.
        x : ndarray, shape(surfs,num_modes,)
            Spectral coefficients in the double Fourier basis.

        """
        s = np.atleast_2d(s)
        c = np.atleast_2d(c)

        M = int(np.max(np.abs(m_0)))
        N = int(np.max(np.abs(n_0)))

        mn_1 = np.array(
            [[m - M, n - N] for m in range(2 * M + 1) for n in range(2 * N + 1)]
        )
        m_1 = mn_1[:, 0]
        n_1 = mn_1[:, 1]
        x = np.zeros((s.shape[0], m_1.size))

        for k in range(len(m_0)):
            # sin(m*theta)*cos(n*phi)
            sin_mn_1 = np.where(
                np.logical_and.reduce((m_1 == -np.abs(m_0[k]), n_1 == np.abs(n_0[k])))
            )[0][0]
            # cos(m*theta)*sin(n*phi)
            sin_mn_2 = np.where(
                np.logical_and.reduce((m_1 == np.abs(m_0[k]), n_1 == -np.abs(n_0[k])))
            )[0][0]
            # cos(m*theta)*cos(n*phi)
            cos_mn_1 = np.where(
                np.logical_and.reduce((m_1 == np.abs(m_0[k]), n_1 == np.abs(n_0[k])))
            )[0][0]
            # sin(m*theta)*sin(n*phi)
            cos_mn_2 = np.where(
                np.logical_and.reduce((m_1 == -np.abs(m_0[k]), n_1 == -np.abs(n_0[k])))
            )[0][0]

            if np.sign(m_0[k]) != 0:
                x[:, sin_mn_1] += s[:, k]
            x[:, cos_mn_1] += c[:, k]
            if np.sign(n_0[k]) > 0:
                x[:, sin_mn_2] -= s[:, k]
                if np.sign(m_0[k]) != 0:
                    x[:, cos_mn_2] += c[:, k]
            elif np.sign(n_0[k]) < 0:
                x[:, sin_mn_2] += s[:, k]
                if np.sign(m_0[k]) != 0:
                    x[:, cos_mn_2] -= c[:, k]

        return m_1, n_1, x

    @staticmethod
    def _ptolemy_identity_rev(m_1, n_1, x):
        """Converts from a double Fourier series of the form:
            ss*sin(m*theta)*sin(n*phi) + sc*sin(m*theta)*cos(n*phi) +
            cs*cos(m*theta)*sin(n*phi) + cc*cos(m*theta)*cos(n*phi)
        to the double-angle form:
            s*sin(m*theta-n*phi) + c*cos(m*theta-n*phi)
        using Ptolemy's sum and difference formulas.

        Parameters
        ----------
        m_1 : ndarray, shape(num_modes,)
            Poloidal mode numbers of the double Fourier basis.
        n_1 : ndarray, shape(num_modes,)
            Toroidal mode numbers of the double Fourier basis.
        x : ndarray, shape(surfs,num_modes,)
            Spectral coefficients in the double Fourier basis.

        Returns
        -------
        m_0 : ndarray
            Poloidal mode numbers of the double-angle Fourier basis.
        n_0 : ndarray
            Toroidal mode numbers of the double-angle Fourier basis.
        s : ndarray, shape(surfs,num_modes)
            Coefficients of sin(m*theta-n*phi) terms.
            Each row is a separate flux surface.
        c : ndarray, shape(surfs,num_modes)
            Coefficients of cos(m*theta-n*phi) terms.
            Each row is a separate flux surface.

        """
        x = np.atleast_2d(x)

        M = int(np.max(np.abs(m_1)))
        N = int(np.max(np.abs(n_1)))

        mn_0 = np.array([[m, n - N] for m in range(M + 1) for n in range(2 * N + 1)])
        m_0 = mn_0[N:, 0]
        n_0 = mn_0[N:, 1]

        s = np.zeros((x.shape[0], m_0.size))
        c = np.zeros_like(s)

        for k in range(len(m_1)):
            # (|m|*theta + |n|*phi)
            idx_pos = np.where(
                np.logical_and.reduce((m_0 == np.abs(m_1[k]), n_0 == -np.abs(n_1[k])))
            )[0]
            # (|m|*theta - |n|*phi)
            idx_neg = np.where(
                np.logical_and.reduce((m_0 == np.abs(m_1[k]), n_0 == np.abs(n_1[k])))
            )[0]

            # if m == 0 and n != 0, p = 0; otherwise p = 1
            p = int(bool(m_1[k])) ** int(bool(n_1[k]))

            if sign(m_1[k]) * sign(n_1[k]) < 0:
                # sin_mn terms
                if idx_pos.size:
                    s[:, idx_pos[0]] += x[:, k] / (2 ** p)
                if idx_neg.size:
                    s[:, idx_neg[0]] += x[:, k] / (2 ** p) * sign(n_1[k])
            else:
                # cos_mn terms
                if idx_pos.size:
                    c[:, idx_pos[0]] += x[:, k] / (2 ** p) * sign(n_1[k])
                if idx_neg.size:
                    c[:, idx_neg[0]] += x[:, k] / (2 ** p)

        return m_0, n_0, s, c

    @staticmethod
    def _fourier_to_zernike(m, n, x_mn, basis: FourierZernikeBasis):
        """Converts from a double Fourier series at each flux surface to a
        Fourier-Zernike basis.

        Parameters
        ----------
        m : ndarray, shape(num_modes,)
            Poloidal mode numbers.
        n : ndarray, shape(num_modes,)
            Toroidal mode numbers.
        x_mn : ndarray, shape(surfs,num_modes)
            Spectral coefficients in the double Fourier basis.
            Each row is a separate flux surface, increasing from the magnetic
            axis to the boundary.
        basis : FourierZernikeBasis
            Basis set for x_lmn

        Returns
        -------
        x_lmn : ndarray, shape(num_modes,)
            Fourier-Zernike spectral coefficients.

        """
        x_lmn = np.zeros((basis.num_modes,))
        surfs = x_mn.shape[0]
        rho = np.sqrt(np.linspace(0, 1, surfs))

        for k in range(len(m)):
            idx = np.where(
                np.logical_and.reduce(
                    (basis.modes[:, 1] == m[k], basis.modes[:, 2] == n[k])
                )
            )[0]
            if len(idx):
                A = jacobi(rho, basis.modes[idx, 0], basis.modes[idx, 1])
                c = np.linalg.lstsq(A, x_mn[:, k], rcond=None)[0]
                x_lmn = put(x_lmn, idx, c)

        return x_lmn

    @staticmethod
    def _zernike_to_fourier(x_lmn, basis: FourierZernikeBasis, rho):
        """Converts from a Fourier-Zernike basis to a double Fourier series at each
        flux surface.

        Parameters
        ----------
        x_lmn : ndarray, shape(num_modes,)
            Fourier-Zernike spectral coefficients.
        basis : FourierZernikeBasis
            Basis set for x_lmn.
        rho : ndarray
            Radial coordinates of flux surfaces, rho = sqrt(psi).

        Returns
        -------
        m : ndarray, shape(num_modes,)
            Poloidal mode numbers.
        n : ndarray, shape(num_modes,)
            Toroidal mode numbers.
        x_mn : ndarray, shape(surfs,num_modes)
            Spectral coefficients in the double Fourier basis.
            Each row is a separate flux surface, increasing from the magnetic
            axis to the boundary.

        """
        M = basis.M
        N = basis.N

        mn = np.array(
            [[m - M, n - N] for m in range(2 * M + 1) for n in range(2 * N + 1)]
        )
        m = mn[:, 0]
        n = mn[:, 1]

        x_mn = np.zeros((rho.size, m.size))
        for k in range(len(m)):
            idx = np.where(
                np.logical_and.reduce(
                    (basis.modes[:, 1] == m[k], basis.modes[:, 2] == n[k])
                )
            )[0]
            if len(idx):
                A = jacobi(rho, basis.modes[idx, 0], basis.modes[idx, 1])
                x_mn[:, k] = np.matmul(A, x_lmn[idx])

        return m, n, x_mn
