import math
import numpy as np
from netCDF4 import Dataset, stringtochar

from desc.backend import put
from desc.utils import Tristate, sign
from desc.grid import LinearGrid, ConcentricGrid
from desc.basis import FourierSeries, DoubleFourierSeries, FourierZernikeBasis, jacobi
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

        # basis symmetry
        if inputs["sym"]:
            R_sym = Tristate(True)
            Z_sym = Tristate(False)
        else:
            R_sym = Tristate(None)
            Z_sym = Tristate(None)

        # profiles
        preset = file.dimensions["preset"].size
        p0 = file.variables["presf"][0] / file.variables["am"][0]
        inputs["profiles"] = np.zeros((preset, 3))
        inputs["profiles"][:, 0] = np.arange(0, 2 * preset, 2)
        inputs["profiles"][:, 1] = file.variables["am"][:] * p0
        inputs["profiles"][:, 2] = file.variables["ai"][:]

        file.close

        # boundary
        m, n, R1_mn = cls._ptolemy_identity_fwd(xm, xn, s=rmns[-1, :], c=rmnc[-1, :])
        m, n, Z1_mn = cls._ptolemy_identity_fwd(xm, xn, s=zmns[-1, :], c=zmnc[-1, :])
        inputs["boundary"] = np.vstack((m, n, R1_mn, Z1_mn)).T

        # axis
        m, n, R0_mn = cls._ptolemy_identity_fwd(xm, xn, s=rmns[0, :], c=rmnc[0, :])
        m, n, Z0_mn = cls._ptolemy_identity_fwd(xm, xn, s=zmns[0, :], c=zmnc[0, :])
        R0_basis = FourierSeries(N=inputs["N"], NFP=inputs["NFP"], sym=R_sym)
        Z0_basis = FourierSeries(N=inputs["N"], NFP=inputs["NFP"], sym=Z_sym)
        inputs["R0_n"] = np.zeros((R0_basis.num_modes,))
        inputs["Z0_n"] = np.zeros((Z0_basis.num_modes,))
        for m, n, R0, Z0 in np.vstack((m, n, R0_mn, Z0_mn)).T:
            idx_R = np.where(
                np.logical_and(R0_basis.modes[:, 1] == m, R0_basis.modes[:, 2] == n)
            )[0]
            idx_Z = np.where(
                np.logical_and(Z0_basis.modes[:, 1] == m, Z0_basis.modes[:, 2] == n)
            )[0]
            inputs["R0_n"] = put(inputs["R0_n"], idx_R, R0)
            inputs["Z0_n"] = put(inputs["Z0_n"], idx_Z, Z0)

        # lambda
        m, n, l_mn = cls._ptolemy_identity_fwd(xm, xn, s=lmns, c=lmnc)
        inputs["l_lmn"], l_basis = cls._fourier_to_zernike(
            m,
            n,
            l_mn,
            NFP=inputs["NFP"],
            L=inputs["L"],
            M=inputs["M"],
            N=inputs["N"],
            index=inputs["index"],
        )

        # grid for fitting r
        grid = ConcentricGrid(
            M=2 * math.ceil(1.5 * inputs["M"]) + 1,
            N=2 * math.ceil(1.5 * inputs["N"]) + 1,
            NFP=inputs["NFP"],
            sym=inputs["sym"],
            axis=True,
            index=inputs["index"],
            surfs="cheb1",
        )

        # initialize Configuration before setting r
        eq = Configuration(inputs=inputs)
        polar_coords = eq.compute_polar_coords(grid)

        # r
        m, n, R_mn = cls._ptolemy_identity_fwd(xm, xn, s=rmns, c=rmnc)
        m, n, Z_mn = cls._ptolemy_identity_fwd(xm, xn, s=zmns, c=zmnc)
        R_lmn, R_basis = cls._fourier_to_zernike(
            m,
            n,
            R_mn,
            NFP=inputs["NFP"],
            L=inputs["L"],
            M=inputs["M"],
            N=inputs["N"],
            index=inputs["index"],
        )
        Z_lmn, Z_basis = cls._fourier_to_zernike(
            m,
            n,
            Z_mn,
            NFP=inputs["NFP"],
            L=inputs["L"],
            M=inputs["M"],
            N=inputs["N"],
            index=inputs["index"],
        )
        R_transform = Transform(grid=grid, basis=R_basis)
        Z_transform = Transform(grid=grid, basis=Z_basis)
        R = R_transform.transform(R_lmn)
        Z = Z_transform.transform(Z_lmn)
        r_R = (R - polar_coords["R0"]) / (polar_coords["R1"] - polar_coords["R0"])
        r_Z = (Z - polar_coords["Z0"]) / (polar_coords["Z1"] - polar_coords["Z0"])
        r = np.where(r_Z <= 1, r_Z, r_R)
        r_transform = Transform(grid=grid, basis=eq.r_basis)
        eq.r_lmn = r_transform.fit(r)

        # apply boundary conditions
        BC = BoundaryConstraint(eq.R0_basis, eq.Z0_basis, eq.r_basis, eq.l_basis)
        eq.x = BC.make_feasible(eq.x)

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

        lasym = file.createVariable("lasym", np.int32, ("dim_00001",))
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

        # spectral data

        MM = 2 * math.ceil(1.5 * M) + 1
        NN = 2 * math.ceil(1.5 * N) + 1
        grid = LinearGrid(M=MM, N=NN, NFP=NFP)

        sin_basis = DoubleFourierSeries(M=M, N=N, NFP=NFP, sym=Tristate(False))
        cos_basis = DoubleFourierSeries(M=M, N=N, NFP=NFP, sym=Tristate(True))
        full_basis = DoubleFourierSeries(M=M, N=N, NFP=NFP, sym=Tristate(None))

        sin_transform = Transform(grid=grid, basis=sin_basis)
        cos_transform = Transform(grid=grid, basis=cos_basis)
        full_transform = Transform(grid=grid, basis=full_basis)

        rmnc = file.createVariable("rmnc", np.float64, ("radius", "mn_mode"))
        rmnc.long_name = "cos(m*t-n*p) component of cylindrical R on full mesh"
        rmnc.units = "m"
        m = cos_basis.modes[:, 1]
        n = cos_basis.modes[:, 2]
        if not eq.sym:
            rmns = file.createVariable("rmns", np.float64, ("radius", "mn_mode"))
            rmns.long_name = "sin(m*t-n*p) component of cylindrical R on full mesh"
            rmns.units = "m"
            m = full_basis.modes[:, 1]
            n = full_basis.modes[:, 2]
        x_mn = np.zeros((surfs, m.size))
        for i in range(surfs):
            grid = LinearGrid(M=MM, N=NN, NFP=NFP, rho=r_full[i])
            data = eq.compute_toroidal_coords(grid)["R"]
            if eq.sym:
                x_mn[i, :] = cos_transform.fit(data)
            else:
                x_mn[i, :] = full_transform.fit(data)
        xm, xn, s, c = cls._ptolemy_identity_rev(m, n, x_mn)
        rmnc[:] = c
        if not eq.sym:
            rmns[:] = s

        zmns = file.createVariable("zmns", np.float64, ("radius", "mn_mode"))
        zmns.long_name = "sin(m*t-n*p) component of cylindrical Z on full mesh"
        zmns.units = "m"
        m = sin_basis.modes[:, 1]
        n = sin_basis.modes[:, 2]
        if not eq.sym:
            zmnc = file.createVariable("zmnc", np.float64, ("radius", "mn_mode"))
            zmnc.long_name = "cos(m*t-n*p) component of cylindrical Z on full mesh"
            zmnc.units = "m"
            m = full_basis.modes[:, 1]
            n = full_basis.modes[:, 2]
        x_mn = np.zeros((surfs, m.size))
        for i in range(surfs):
            grid = LinearGrid(M=MM, N=NN, NFP=NFP, rho=r_full[i])
            data = eq.compute_toroidal_coords(grid)["Z"]
            if eq.sym:
                x_mn[i, :] = sin_transform.fit(data)
            else:
                x_mn[i, :] = full_transform.fit(data)
        xm, xn, s, c = cls._ptolemy_identity_rev(m, n, x_mn)
        zmns[:] = s
        if not eq.sym:
            zmnc[:] = c

        lmns = file.createVariable("lmns", np.float64, ("radius", "mn_mode"))
        lmns.long_name = "sin(m*t-n*p) component of lambda on full mesh"
        lmns.units = "m"
        m = sin_basis.modes[:, 1]
        n = sin_basis.modes[:, 2]
        if not eq.sym:
            lmnc = file.createVariable("zmnc", np.float64, ("radius", "mn_mode"))
            lmnc.long_name = "cos(m*t-n*p) component of lambda on full mesh"
            lmnc.units = "m"
            m = full_basis.modes[:, 1]
            n = full_basis.modes[:, 2]
        x_mn = np.zeros((surfs, m.size))
        for i in range(surfs):
            grid = LinearGrid(M=MM, N=NN, NFP=NFP, rho=r_full[i])
            data = eq.compute_polar_coords(grid)["lambda"]
            if eq.sym:
                x_mn[i, :] = sin_transform.fit(data)
            else:
                x_mn[i, :] = full_transform.fit(data)
        xm, xn, s, c = cls._ptolemy_identity_rev(m, n, x_mn)
        lmns[:] = s
        if not eq.sym:
            lmnc[:] = c

        file.close

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
        s : ndarray, optional
            Coefficients of sin(m*theta-n*phi) terms.
            Each row is a separate flux surface.
        c : ndarray, optional
            Coefficients of cos(m*theta-n*phi) terms.
            Each row is a separate flux surface.

        Returns
        -------
        m_1 : ndarray, shape(num_modes,)
            Poloidal mode numbers of the double Fourier basis.
        n_1 : ndarray, shape(num_modes,)
            Toroidal mode numbers of the double Fourier basis.
        x : ndarray, shape(num_modes,)
            Spectral coefficients in the double Fourier basis.

        """
        s = np.atleast_2d(s)
        c = np.atleast_2d(c)

        M = int(np.max(np.abs(m_0)))
        N = int(np.max(np.abs(n_0)))

        mn_1 = np.array(
            [
                [m_0 - M, n_0 - N, 0]
                for m_0 in range(2 * M + 1)
                for n_0 in range(2 * N + 1)
            ]
        )
        m_1 = mn_1[:, 0]
        n_1 = mn_1[:, 1]
        x = np.zeros((s.shape[0], m_1.size))

        for i in range(len(m_0)):
            # sin(m*theta)*cos(n*phi)
            sin_mn_1 = np.where(
                np.logical_and(m_1 == -np.abs(m_0[i]), n_1 == np.abs(n_0[i]))
            )[0][0]
            # cos(m*theta)*sin(n*phi)
            sin_mn_2 = np.where(
                np.logical_and(m_1 == np.abs(m_0[i]), n_1 == -np.abs(n_0[i]))
            )[0][0]
            # cos(m*theta)*cos(n*phi)
            cos_mn_1 = np.where(
                np.logical_and(m_1 == np.abs(m_0[i]), n_1 == np.abs(n_0[i]))
            )[0][0]
            # sin(m*theta)*sin(n*phi)
            cos_mn_2 = np.where(
                np.logical_and(m_1 == -np.abs(m_0[i]), n_1 == -np.abs(n_0[i]))
            )[0][0]

            if np.sign(m_0[i]) != 0:
                x[:, sin_mn_1] += s[:, i]
            x[:, cos_mn_1] += c[:, i]
            if np.sign(n_0[i]) > 0:
                x[:, sin_mn_2] -= s[:, i]
                if np.sign(m_0[i]) != 0:
                    x[:, cos_mn_2] += c[:, i]
            elif np.sign(n_0[i]) < 0:
                x[:, sin_mn_2] += s[:, i]
                if np.sign(m_0[i]) != 0:
                    x[:, cos_mn_2] -= c[:, i]

        return m_1, n_1, x

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
        x : ndarray, shape(num_modes,)
            Spectral coefficients in the double Fourier basis.

        Returns
        -------
        m_0 : ndarray
            Poloidal mode numbers of the double-angle Fourier basis.
        n_0 : ndarray
            Toroidal mode numbers of the double-angle Fourier basis.
        s : ndarray, optional
            Coefficients of sin(m*theta-n*phi) terms.
            Each row is a separate flux surface.
        c : ndarray, optional
            Coefficients of cos(m*theta-n*phi) terms.
            Each row is a separate flux surface.

        """
        x = np.atleast_2d(x)

        m_0 = np.abs(m_1).flatten()
        n_0 = n_1.flatten()
        sort_idx = np.lexsort((n_0, m_0))
        m_0 = m_0[sort_idx]
        n_0 = n_0[sort_idx]
        if m_0[0] != 0 or n_0[0] != 0:
            m_0 = np.insert(m_0, 0, 0)
            n_0 = np.insert(n_0, 0, 0)
        n_0 = np.where(m_0 == 0, np.abs(n_0), n_0)

        s = np.zeros((x.shape[0], m_0.size))
        c = np.zeros_like(s)

        for i in range(len(m_1)):
            # (|m|*theta + |n|*phi)
            idx_pos = np.where(
                np.logical_and(m_0 == np.abs(m_1[i]), n_0 == -np.abs(n_1[i]))
            )[0]
            # (|m|*theta - |n|*phi)
            idx_neg = np.where(
                np.logical_and(m_0 == np.abs(m_1[i]), n_0 == np.abs(n_1[i]))
            )[0]

            if sign(m_1[i]) * sign(n_1[i]) < 0:
                # sin_mn terms
                if idx_pos.size:
                    s[:, idx_pos[0]] += x[:, i] / 2
                if idx_neg.size:
                    s[:, idx_neg[0]] += x[:, i] / 2 * sign(n_1[i])
            else:
                # cos_mn terms
                if idx_pos.size:
                    c[:, idx_pos[0]] += x[:, i] / 2 * sign(n_1[i])
                if idx_neg.size:
                    c[:, idx_neg[0]] += x[:, i] / 2

        return m_0, n_0, s, c

    def _fourier_to_zernike(
        m,
        n,
        x_mn,
        NFP: int = 1,
        L: int = -1,
        M: int = -1,
        N: int = -1,
        index: str = "ansi",
    ):
        """Converts from a double Fourier series at each flux surface to a
        Fourier-Zernike basis.

        Parameters
        ----------
        m : ndarray, shape(num_modes,)
            Poloidal mode numbers.
        n : ndarray, shape(num_modes,)
            Toroidal mode numbers.
        x_mn : ndarray, shape(num_modes,)
            Spectral coefficients in the double Fourier basis.
            Each row is a separate flux surface, increasing from the magnetic
            axis to the boundary.
        NFP : int, optional
            Number of toroidal field periods.
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
        x_lmn : ndarray, shape(num_modes,)
            Fourier-Zernike spectral coefficients.
        basis : FourierZernikeBasis
            Basis set for x_lmn

        """
        M = M if M > 0 else int(np.max(np.abs(m)))
        N = N if N >= 0 else int(np.max(np.abs(n)))

        if not np.any(x_mn[:, np.where(sign(m) * sign(n) == -1)[0]]):
            sym = Tristate(True)
        elif not np.any(x_mn[:, np.where(sign(m) * sign(n) == 1)[0]]):
            sym = Tristate(False)
        else:
            sym = Tristate(None)

        basis = FourierZernikeBasis(L=L, M=M, N=N, NFP=NFP, sym=sym, index=index)
        x_lmn = np.zeros((basis.num_modes,))

        surfs = x_mn.shape[0]
        rho = np.sqrt(np.linspace(0, 1, surfs))

        for i in range(len(m)):
            idx = np.where(
                np.logical_and(basis.modes[:, 1] == m[i], basis.modes[:, 2] == n[i])
            )[0]
            if len(idx):
                A = jacobi(rho, basis.modes[idx, 0], basis.modes[idx, 1])
                c = np.linalg.lstsq(A, x_mn[:, i], rcond=None)[0]
                x_lmn = put(x_lmn, idx, c)

        return x_lmn, basis


def vmec_transf(xmna, xm, xn, theta, phi, trig="sin"):
    """Compute Fourier transform of VMEC data

    Parameters
    ----------
    xmns : 2d float array
        xmnc[:,i] are the sin coefficients at flux surface i
    xm : 1d int array
        poloidal mode numbers
    xn : 1d int array
        toroidal mode numbers
    theta : 1d float array
        poloidal angles
    phi : 1d float array
        toroidal angles
    trig : string
        type of transform, options are 'sin' or 'cos' (Default value = 'sin')
    xmna :


    Returns
    -------
    f : ndarray
        f[i,j,k] is the transformed data at flux surface i, theta[j], phi[k]

    """

    ns = np.shape(np.atleast_2d(xmna))[0]
    lt = np.size(theta)
    lp = np.size(phi)
    # Create mode x angle arrays
    mtheta = np.atleast_2d(xm).T @ np.atleast_2d(theta)
    nphi = np.atleast_2d(xn).T @ np.atleast_2d(phi)
    # Create trig arrays
    cosmt = np.cos(mtheta)
    sinmt = np.sin(mtheta)
    cosnp = np.cos(nphi)
    sinnp = np.sin(nphi)
    # Calcualte the transform
    f = np.zeros((ns, lt, lp))
    for k in range(ns):
        xmn = np.tile(np.atleast_2d(np.atleast_2d(xmna)[k, :]).T, (1, lt))
        if trig == "sin":
            f[k, :, :] = np.tensordot((xmn * sinmt).T, cosnp, axes=1) + np.tensordot(
                (xmn * cosmt).T, sinnp, axes=1
            )
        elif trig == "cos":
            f[k, :, :] = np.tensordot((xmn * cosmt).T, cosnp, axes=1) - np.tensordot(
                (xmn * sinmt).T, sinnp, axes=1
            )
    return f
