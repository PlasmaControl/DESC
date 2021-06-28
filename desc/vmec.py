import os
import numpy as np

from netCDF4 import Dataset, stringtochar
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from scipy import optimize, interpolate, integrate

from desc.utils import Timer, sign
from desc.grid import Grid, LinearGrid
from desc.basis import DoubleFourierSeries
from desc.transform import Transform
from desc.profiles import PowerSeriesProfile
from desc.equilibrium import Equilibrium
from desc.boundary_conditions import LCFSConstraint
from desc.vmec_utils import (
    ptolemy_identity_fwd,
    ptolemy_identity_rev,
    fourier_to_zernike,
    zernike_to_fourier,
)


class VMECIO:
    """Performs input from VMEC netCDF files to DESC Equilibrium and vice-versa."""

    @classmethod
    def load(cls, path, L=-1, M=-1, N=-1, spectral_indexing="ansi"):
        """Load a VMEC netCDF file as a Equilibrium.

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
        spectral_indexing : str, optional
            Type of Zernike indexing scheme to use. (Default = ``'ansi'``)

        Returns
        -------
        eq: Equilibrium
            Equilibrium that resembles the VMEC data.

        """
        file = Dataset(path, mode="r")
        inputs = {}

        # parameters
        inputs["Psi"] = float(file.variables["phi"][-1])
        inputs["NFP"] = int(file.variables["nfp"][0])
        inputs["M"] = M if M > 0 else int(file.variables["mpol"][0] - 1)
        inputs["N"] = N if N >= 0 else int(file.variables["ntor"][0])
        inputs["spectral_indexing"] = spectral_indexing
        default_L = {
            "ansi": inputs["M"],
            "fringe": 2 * inputs["M"],
            "chevron": inputs["M"],
            "house": 2 * inputs["M"],
        }
        inputs["L"] = L if L >= 0 else default_L[inputs["spectral_indexing"]]

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
        m, n, Rb_lmn = ptolemy_identity_fwd(xm, xn, s=rmns[-1, :], c=rmnc[-1, :])
        m, n, Zb_lmn = ptolemy_identity_fwd(xm, xn, s=zmns[-1, :], c=zmnc[-1, :])
        inputs["boundary"] = np.vstack((np.zeros_like(m), m, n, Rb_lmn, Zb_lmn)).T

        # initialize Equilibrium
        eq = Equilibrium(inputs=inputs)

        # R
        m, n, R_mn = ptolemy_identity_fwd(xm, xn, s=rmns, c=rmnc)
        eq.R_lmn = fourier_to_zernike(m, n, R_mn, eq.R_basis)

        # Z
        m, n, Z_mn = ptolemy_identity_fwd(xm, xn, s=zmns, c=zmnc)
        eq.Z_lmn = fourier_to_zernike(m, n, Z_mn, eq.Z_basis)

        # lambda
        m, n, L_mn = ptolemy_identity_fwd(xm, xn, s=lmns, c=lmnc)
        eq.L_lmn = fourier_to_zernike(m, n, L_mn, eq.L_basis)

        # apply boundary conditions
        BC = eq.surface.get_constraint(
            eq.R_basis,
            eq.Z_basis,
            eq.L_basis,
        )
        eq.x = BC.make_feasible(eq.x)

        return eq

    @classmethod
    def save(cls, eq, path, surfs=128, verbose=1):
        """Save an Equilibrium as a netCDF file in the VMEC format.

        Parameters
        ----------
        eq : Equilibrium
            Equilibrium to save.
        path : str
            File path of output data.
        surfs: int
            Number of flux surfaces to interpolate at (Default = 128).
        verbose: int
            Level of output (Default = 1).
            * 0: no output
            * 1: status of quantities computed
            * 2: as above plus timing information

        Returns
        -------
        None

        """
        timer = Timer()
        timer.start("Total time")

        """ VMEC netCDF file is generated in VMEC2000/Sources/Input_Output/wrout.f
            see lines 300+ for full list of included variables
        """
        file = Dataset(path, mode="w", format="NETCDF3_64BIT_OFFSET")

        Psi = eq.Psi
        NFP = eq.NFP
        M = eq.M
        N = eq.N
        M_nyq = M + 4
        N_nyq = N + 2 if N > 0 else 0

        # VMEC radial coordinate: s = rho^2 = Psi / Psi(LCFS)
        s_full = np.linspace(0, 1, surfs)
        s_half = s_full[0:-1] + 0.5 / (surfs - 1)
        r_full = np.sqrt(s_full)
        r_half = np.sqrt(s_half)

        # dimensions
        file.createDimension("radius", surfs)  # number of flux surfaces
        file.createDimension(
            "mn_mode", (2 * N + 1) * M + N + 1
        )  # number of Fourier modes
        file.createDimension(
            "mn_mode_nyq", (2 * N_nyq + 1) * M_nyq + N_nyq + 1
        )  # number of Nyquist Fourier modes
        file.createDimension("n_tor", N + 1)  # number of toroidal Fourier modes
        file.createDimension("preset", 21)  # dimension of profile inputs
        file.createDimension("ndfmax", 101)  # used for am_aux & ai_aux
        file.createDimension("time", 100)  # used for fsqrt & wdot
        file.createDimension("dim_00001", 1)
        file.createDimension("dim_00020", 20)
        file.createDimension("dim_00100", 100)
        file.createDimension("dim_00200", 200)
        preset = file.dimensions["preset"].size

        # parameters
        timer.start("parameters")
        if verbose > 0:
            print("Saving parameters")

        version_ = file.createVariable("version_", np.float64)
        version_[:] = 9  # VMEC version at the time of this writing

        input_extension = file.createVariable("input_extension", "S1", ("dim_00100",))
        input_extension[:] = stringtochar(
            np.array([" " * 100], "S" + str(file.dimensions["dim_00100"].size))
        )  # VMEC input filename: input.[input_extension]

        mgrid_mode = file.createVariable("mgrid_mode", "S1", ("dim_00001",))
        mgrid_mode[:] = stringtochar(
            np.array([""], "S" + str(file.dimensions["dim_00001"].size))
        )

        mgrid_file = file.createVariable("mgrid_file", "S1", ("dim_00200",))
        mgrid_file[:] = stringtochar(
            np.array(["none" + " " * 196], "S" + str(file.dimensions["dim_00200"].size))
        )

        ier_flag = file.createVariable("ier_flag", np.int32)
        ier_flag.long_name = "error flag (0 = solved equilibrium, 1 = unsolved)"
        ier_flag[:] = int(not eq.solved)

        lfreeb = file.createVariable("lfreeb__logical__", np.int32)
        lfreeb.long_name = "free boundary logical (0 = fixed boundary)"
        lfreeb[:] = 0

        lrecon = file.createVariable("lrecon__logical__", np.int32)
        lrecon.long_name = "reconstruction logical (0 = no reconstruction)"
        lrecon[:] = 0

        lrfp = file.createVariable("lrfp__logical__", np.int32)
        lrfp.long_name = "reverse-field pinch logical (0 = not an RFP)"
        lrfp[:] = 0

        lasym = file.createVariable("lasym__logical__", np.int32)
        lasym.long_name = "asymmetry logical (0 = stellarator symmetry)"
        lasym[:] = int(not eq.sym)

        nfp = file.createVariable("nfp", np.int32)
        nfp.long_name = "number of field periods"
        nfp[:] = NFP

        ns = file.createVariable("ns", np.int32)
        ns.long_name = "number of flux surfaces"
        ns[:] = surfs

        mpol = file.createVariable("mpol", np.int32)
        mpol.long_name = "number of poloidal Fourier modes"
        mpol[:] = M + 1

        ntor = file.createVariable("ntor", np.int32)
        ntor.long_name = "number of positive toroidal Fourier modes"
        ntor[:] = N

        mnmax = file.createVariable("mnmax", np.int32)
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

        mnmax_nyq = file.createVariable("mnmax_nyq", np.int32)
        mnmax_nyq.long_name = "total number of Nyquist Fourier modes"
        mnmax_nyq[:] = file.dimensions["mn_mode_nyq"].size

        xm_nyq = file.createVariable("xm_nyq", np.float64, ("mn_mode_nyq",))
        xm_nyq.long_name = "poloidal Nyquist mode numbers"
        xm_nyq[:] = np.tile(
            np.linspace(0, M_nyq, M_nyq + 1), (2 * N_nyq + 1, 1)
        ).T.flatten()[-file.dimensions["mn_mode_nyq"].size :]

        xn_nyq = file.createVariable("xn_nyq", np.float64, ("mn_mode_nyq",))
        xn_nyq.long_name = "toroidal Nyquist mode numbers"
        xn_nyq[:] = np.tile(np.linspace(-N_nyq, N_nyq, 2 * N_nyq + 1) * NFP, M_nyq + 1)[
            -file.dimensions["mn_mode_nyq"].size :
        ]

        signgs = file.createVariable("signgs", np.int32)
        signgs.long_name = "sign of coordinate system jacobian"
        signgs[:] = sign(eq.compute_jacobian(Grid(np.array([[1, 0, 0]])))["g"])

        gamma = file.createVariable("gamma", np.float64)
        gamma.long_name = "compressibility index (0 = pressure prescribed)"
        gamma[:] = 0

        # TODO: add option for saving spline profiles
        power_series = stringtochar(
            np.array(
                ["power_series" + " " * 8], "S" + str(file.dimensions["dim_00020"].size)
            )
        )

        pmass_type = file.createVariable("pmass_type", "S1", ("dim_00020",))
        pmass_type.long_name = "parameterization of pressure function"
        pmass_type[:] = power_series

        piota_type = file.createVariable("piota_type", "S1", ("dim_00020",))
        piota_type.long_name = "parameterization of rotational transform function"
        piota_type[:] = power_series

        pcurr_type = file.createVariable("pcurr_type", "S1", ("dim_00020",))
        pcurr_type.long_name = "parameterization of current density function"
        pcurr_type[:] = power_series

        am = file.createVariable("am", np.float64, ("preset",))
        am.long_name = "pressure coefficients"
        am.units = "Pa"
        am[:] = np.zeros((file.dimensions["preset"].size,))
        # only using up to 10th order to avoid poor conditioning
        am[:11] = PowerSeriesProfile.from_values(
            s_full, eq.pressure(r_full), order=10).params

        ai = file.createVariable("ai", np.float64, ("preset",))
        ai.long_name = "rotational transform coefficients"
        ai[:] = np.zeros((file.dimensions["preset"].size,))
        # only using up to 10th order to avoid poor conditioning
        ai[:11] = PowerSeriesProfile.from_values(
            s_full, eq.iota(r_full), order=10).params

        ac = file.createVariable("ac", np.float64, ("preset",))
        ac.long_name = "normalized toroidal current density coefficients"
        ac[:] = np.zeros((file.dimensions["preset"].size,))

        presf = file.createVariable("presf", np.float64, ("radius",))
        presf.long_name = "pressure on full mesh"
        presf.units = "Pa"
        presf[:] = eq.pressure(r_full)

        pres = file.createVariable("pres", np.float64, ("radius",))
        pres.long_name = "pressure on half mesh"
        pres.units = "Pa"
        pres[0] = 0
        pres[1:] = eq.pressure(r_half)

        mass = file.createVariable("mass", np.float64, ("radius",))
        mass.long_name = "mass on half mesh"
        mass.units = "Pa"
        mass[:] = pres[:]

        iotaf = file.createVariable("iotaf", np.float64, ("radius",))
        iotaf.long_name = "rotational transform on full mesh"
        iotaf[:] = eq.iota(r_full)

        iotas = file.createVariable("iotas", np.float64, ("radius",))
        iotas.long_name = "rotational transform on half mesh"
        iotas[0] = 0
        iotas[1:] = eq.iota(r_half)

        phi = file.createVariable("phi", np.float64, ("radius",))
        phi.long_name = "toroidal flux"
        phi.units = "Wb"
        phi[:] = np.linspace(0, Psi, surfs)

        phipf = file.createVariable("phipf", np.float64, ("radius",))
        phipf.long_name = "d(phi)/ds: toroidal flux derivative"
        phipf[:] = Psi * np.ones((surfs,))

        phips = file.createVariable("phips", np.float64, ("radius",))
        phips.long_name = "d(phi)/ds * -1/2pi: toroidal flux derivative on half mesh"
        phips[0] = 0
        phips[1:] = -phipf[1:] / (2 * np.pi)

        chi = file.createVariable("chi", np.float64, ("radius",))
        chi.long_name = "poloidal flux"
        chi.units = "Wb"
        chi[:] = 2 * Psi * integrate.cumtrapz(r_full * iotaf[:], r_full, initial=0)

        chipf = file.createVariable("chipf", np.float64, ("radius",))
        chipf.long_name = "d(chi)/ds: poloidal flux derivative"
        chipf[:] = phipf[:] * iotaf[:]

        Rmajor_p = file.createVariable("Rmajor_p", np.float64)
        Rmajor_p.long_name = "major radius"
        Rmajor_p.units = "m"
        Rmajor_p[:] = eq.major_radius

        Aminor_p = file.createVariable("Aminor_p", np.float64)
        Aminor_p.long_name = "minor radius"
        Aminor_p.units = "m"
        Aminor_p[:] = eq.minor_radius

        aspect = file.createVariable("aspect", np.float64)
        aspect.long_name = "aspect ratio = R_major / A_minor"
        aspect[:] = eq.aspect_ratio

        volume_p = file.createVariable("volume_p", np.float64)
        volume_p.long_name = "plasma volume"
        volume_p.units = "m^3"
        volume_p[:] = eq.compute_volume()

        timer.stop("parameters")
        if verbose > 1:
            timer.disp("parameters")

        # indepentent variables (exact conversion)

        # R axis
        idx = np.where(eq.R_basis.modes[:, 1] == 0)[0]
        R0_n = np.zeros((2 * N + 1,))
        for k in idx:
            (l, m, n) = eq.R_basis.modes[k, :]
            R0_n[n + N] += (-2 * (l % 2) + 1) * eq.R_lmn[k]
        raxis_cc = file.createVariable("raxis_cc", np.float64, ("n_tor",))
        raxis_cc.long_name = "cos(n*p) component of magnetic axis R coordinate"
        raxis_cc[:] = R0_n[N:]
        if not eq.sym:
            raxis_cs = file.createVariable("raxis_cs", np.float64, ("n_tor",))
            raxis_cs.long_name = "sin(n*p) component of magnetic axis R coordinate"
            raxis_cs[1:] = R0_n[0:N]

        # Z axis
        idx = np.where(eq.Z_basis.modes[:, 1] == 0)[0]
        Z0_n = np.zeros((2 * N + 1,))
        for k in idx:
            (l, m, n) = eq.Z_basis.modes[k, :]
            Z0_n[n + N] += (-2 * (l % 2) + 1) * eq.Z_lmn[k]
        zaxis_cs = file.createVariable("zaxis_cs", np.float64, ("n_tor",))
        zaxis_cs.long_name = "sin(n*p) component of magnetic axis Z coordinate"
        zaxis_cs[1:] = Z0_n[0:N]
        if not eq.sym:
            zaxis_cc = file.createVariable("zaxis_cc", np.float64, ("n_tor",))
            zaxis_cc.long_name = "cos(n*p) component of magnetic axis Z coordinate"
            zaxis_cc[1:] = Z0_n[N:]

        # R
        timer.start("R")
        if verbose > 0:
            print("Saving R")
        rmnc = file.createVariable("rmnc", np.float64, ("radius", "mn_mode"))
        rmnc.long_name = "cos(m*t-n*p) component of cylindrical R, on full mesh"
        rmnc.units = "m"
        if not eq.sym:
            rmns = file.createVariable("rmns", np.float64, ("radius", "mn_mode"))
            rmns.long_name = "sin(m*t-n*p) component of cylindrical R, on full mesh"
            rmns.units = "m"
        m, n, x_mn = zernike_to_fourier(eq.R_lmn, basis=eq.R_basis, rho=r_full)
        xm, xn, s, c = ptolemy_identity_rev(m, n, x_mn)
        rmnc[:] = c
        if not eq.sym:
            rmns[:] = s
        timer.stop("R")
        if verbose > 1:
            timer.disp("R")

        # Z
        timer.start("Z")
        if verbose > 0:
            print("Saving Z")
        zmns = file.createVariable("zmns", np.float64, ("radius", "mn_mode"))
        zmns.long_name = "sin(m*t-n*p) component of cylindrical Z, on full mesh"
        zmns.units = "m"
        if not eq.sym:
            zmnc = file.createVariable("zmnc", np.float64, ("radius", "mn_mode"))
            zmnc.long_name = "cos(m*t-n*p) component of cylindrical Z, on full mesh"
            zmnc.units = "m"
        m, n, x_mn = zernike_to_fourier(eq.Z_lmn, basis=eq.Z_basis, rho=r_full)
        xm, xn, s, c = ptolemy_identity_rev(m, n, x_mn)
        zmns[:] = s
        if not eq.sym:
            zmnc[:] = c
        timer.stop("Z")
        if verbose > 1:
            timer.disp("Z")

        # lambda
        timer.start("lambda")
        if verbose > 0:
            print("Saving lambda")
        lmns = file.createVariable("lmns", np.float64, ("radius", "mn_mode"))
        lmns.long_name = "sin(m*t-n*p) component of lambda, on half mesh"
        lmns.units = "rad"
        if not eq.sym:
            lmnc = file.createVariable("lmnc", np.float64, ("radius", "mn_mode"))
            lmnc.long_name = "cos(m*t-n*p) component of lambda, on half mesh"
            lmnc.units = "rad"
        m, n, x_mn = zernike_to_fourier(eq.L_lmn, basis=eq.L_basis, rho=r_half)
        xm, xn, s, c = ptolemy_identity_rev(m, n, x_mn)
        lmns[0, :] = 0
        lmns[1:, :] = s
        if not eq.sym:
            lmnc[0, :] = 0
            lmnc[1:, :] = c
        timer.stop("lambda")
        if verbose > 1:
            timer.disp("lambda")

        # derived quantities (approximate conversion)

        grid = LinearGrid(M=2 * M_nyq + 1, N=2 * N_nyq + 1, NFP=NFP, rho=1)
        coords = eq.compute_toroidal_coords(grid)
        if eq.sym:
            sin_basis = DoubleFourierSeries(M=M_nyq, N=N_nyq, NFP=NFP, sym="sin")
            cos_basis = DoubleFourierSeries(M=M_nyq, N=N_nyq, NFP=NFP, sym="cos")
            sin_transform = Transform(grid=grid, basis=sin_basis, build_pinv=True)
            cos_transform = Transform(grid=grid, basis=cos_basis, build_pinv=True)
        else:
            full_basis = DoubleFourierSeries(M=M_nyq, N=N_nyq, NFP=NFP, sym=None)
            full_transform = Transform(grid=grid, basis=full_basis, build_pinv=True)

        rmin_surf = file.createVariable("rmin_surf", np.float64)
        rmin_surf.long_name = "minimum R coordinate range"
        rmin_surf.units = "m"
        rmin_surf[:] = np.amin(coords["R"])

        rmax_surf = file.createVariable("rmax_surf", np.float64)
        rmax_surf.long_name = "maximum R coordinate range"
        rmax_surf.units = "m"
        rmax_surf[:] = np.amax(coords["R"])

        zmax_surf = file.createVariable("zmax_surf", np.float64)
        zmax_surf.long_name = "maximum Z coordinate range"
        zmax_surf.units = "m"
        zmax_surf[:] = np.amax(np.abs(coords["Z"]))

        # Jacobian
        timer.start("Jacobian")
        if verbose > 0:
            print("Saving Jacobian")
        gmnc = file.createVariable("gmnc", np.float64, ("radius", "mn_mode_nyq"))
        gmnc.long_name = "cos(m*t-n*p) component of Jacobian, on half mesh"
        gmnc.units = "m"
        m = cos_basis.modes[:, 1]
        n = cos_basis.modes[:, 2]
        if not eq.sym:
            gmns = file.createVariable("gmns", np.float64, ("radius", "mn_mode_nyq"))
            gmns.long_name = "sin(m*t-n*p) component of Jacobian, on half mesh"
            gmns.units = "m"
            m = full_basis.modes[:, 1]
            n = full_basis.modes[:, 2]
        x_mn = np.zeros((surfs - 1, m.size))
        for k in range(surfs - 1):
            grid = LinearGrid(M=2 * M_nyq + 1, N=2 * N_nyq + 1, NFP=NFP, rho=r_half[k])
            data = eq.compute_jacobian(grid)["g"]
            if eq.sym:
                x_mn[k, :] = cos_transform.fit(data)
            else:
                x_mn[k, :] = full_transform.fit(data)
        xm, xn, s, c = ptolemy_identity_rev(m, n, x_mn)
        gmnc[0, :] = 0
        gmnc[1:, :] = c
        if not eq.sym:
            gmns[0, :] = 0
            gmns[1:, :] = s
        timer.stop("Jacobian")
        if verbose > 1:
            timer.disp("Jacobian")

        # |B|
        timer.start("|B|")
        if verbose > 0:
            print("Saving |B|")
        bmnc = file.createVariable("bmnc", np.float64, ("radius", "mn_mode_nyq"))
        bmnc.long_name = "cos(m*t-n*p) component of |B|, on half mesh"
        bmnc.units = "T"
        m = cos_basis.modes[:, 1]
        n = cos_basis.modes[:, 2]
        if not eq.sym:
            bmns = file.createVariable("bmns", np.float64, ("radius", "mn_mode_nyq"))
            bmns.long_name = "sin(m*t-n*p) component of |B|, on half mesh"
            bmns.units = "T"
            m = full_basis.modes[:, 1]
            n = full_basis.modes[:, 2]
        x_mn = np.zeros((surfs - 1, m.size))
        for k in range(surfs - 1):
            grid = LinearGrid(M=2 * M_nyq + 1, N=2 * N_nyq + 1, NFP=NFP, rho=r_half[k])
            data = eq.compute_magnetic_field(grid)["|B|"]
            if eq.sym:
                x_mn[k, :] = cos_transform.fit(data)
            else:
                x_mn[k, :] = full_transform.fit(data)
        xm, xn, s, c = ptolemy_identity_rev(m, n, x_mn)
        bmnc[0, :] = 0
        bmnc[1:, :] = c
        if not eq.sym:
            bmns[0, :] = 0
            bmns[1:, :] = s
        timer.stop("|B|")
        if verbose > 1:
            timer.disp("|B|")

        # B^theta
        timer.start("B^theta")
        if verbose > 0:
            print("Saving B^theta")
        bsupumnc = file.createVariable(
            "bsupumnc", np.float64, ("radius", "mn_mode_nyq")
        )
        bsupumnc.long_name = "cos(m*t-n*p) component of B^theta, on half mesh"
        bsupumnc.units = "T/m"
        m = cos_basis.modes[:, 1]
        n = cos_basis.modes[:, 2]
        if not eq.sym:
            bsupumns = file.createVariable(
                "bsupumns", np.float64, ("radius", "mn_mode_nyq")
            )
            bsupumns.long_name = "sin(m*t-n*p) component of B^theta, on half mesh"
            bsupumns.units = "T/m"
            m = full_basis.modes[:, 1]
            n = full_basis.modes[:, 2]
        x_mn = np.zeros((surfs - 1, m.size))
        for k in range(surfs - 1):
            grid = LinearGrid(M=2 * M_nyq + 1, N=2 * N_nyq + 1, NFP=NFP, rho=r_half[k])
            data = eq.compute_magnetic_field(grid)["B^theta"]
            if eq.sym:
                x_mn[k, :] = cos_transform.fit(data)
            else:
                x_mn[k, :] = full_transform.fit(data)
        xm, xn, s, c = ptolemy_identity_rev(m, n, x_mn)
        bsupumnc[0, :] = 0
        bsupumnc[1:, :] = c
        if not eq.sym:
            bsupumns[0, :] = 0
            bsupumns[1:, :] = s
        timer.stop("B^theta")
        if verbose > 1:
            timer.disp("B^theta")

        # B^zeta
        timer.start("B^zeta")
        if verbose > 0:
            print("Saving B^zeta")
        bsupvmnc = file.createVariable(
            "bsupvmnc", np.float64, ("radius", "mn_mode_nyq")
        )
        bsupvmnc.long_name = "cos(m*t-n*p) component of B^zeta, on half mesh"
        bsupvmnc.units = "T/m"
        m = cos_basis.modes[:, 1]
        n = cos_basis.modes[:, 2]
        if not eq.sym:
            bsupvmns = file.createVariable(
                "bsupvmns", np.float64, ("radius", "mn_mode_nyq")
            )
            bsupvmns.long_name = "sin(m*t-n*p) component of B^zeta, on half mesh"
            bsupvmns.units = "T/m"
            m = full_basis.modes[:, 1]
            n = full_basis.modes[:, 2]
        x_mn = np.zeros((surfs - 1, m.size))
        for k in range(surfs - 1):
            grid = LinearGrid(M=2 * M_nyq + 1, N=2 * N_nyq + 1, NFP=NFP, rho=r_half[k])
            data = eq.compute_magnetic_field(grid)["B^zeta"]
            if eq.sym:
                x_mn[k, :] = cos_transform.fit(data)
            else:
                x_mn[k, :] = full_transform.fit(data)
        xm, xn, s, c = ptolemy_identity_rev(m, n, x_mn)
        bsupvmnc[0, :] = 0
        bsupvmnc[1:, :] = c
        if not eq.sym:
            bsupvmns[0, :] = 0
            bsupvmns[1:, :] = s
        timer.stop("B^zeta")
        if verbose > 1:
            timer.disp("B^zeta")

        # B_psi
        timer.start("B_psi")
        if verbose > 0:
            print("Saving B_psi")
        bsubsmns = file.createVariable(
            "bsubsmns", np.float64, ("radius", "mn_mode_nyq")
        )
        bsubsmns.long_name = "sin(m*t-n*p) component of B_psi, on full mesh"
        bsubsmns.units = "T*m"
        m = sin_basis.modes[:, 1]
        n = sin_basis.modes[:, 2]
        if not eq.sym:
            bsubsmnc = file.createVariable(
                "bsubsmnc", np.float64, ("radius", "mn_mode_nyq")
            )
            bsubsmnc.long_name = "cos(m*t-n*p) component of B_psi, on full mesh"
            bsubsmnc.units = "T*m"
            m = full_basis.modes[:, 1]
            n = full_basis.modes[:, 2]
        x_mn = np.zeros((surfs, m.size))
        for k in range(surfs):
            grid = LinearGrid(M=2 * M_nyq + 1, N=2 * N_nyq + 1, NFP=NFP, rho=r_full[k])
            data = eq.compute_magnetic_field(grid)["B_rho"] / (2 * r_full[k])
            # B_rho -> B_psi conversion: d(rho)/d(s) = 1/(2*rho)
            if eq.sym:
                x_mn[k, :] = sin_transform.fit(data)
            else:
                x_mn[k, :] = full_transform.fit(data)
        xm, xn, s, c = ptolemy_identity_rev(m, n, x_mn)
        bsubsmns[:, :] = s
        bsubsmns[0, :] = (  # linear extrapolation for coefficient at the magnetic axis
            s[1, :] - (s[2, :] - s[1, :]) / (s_full[2] - s_full[1]) * s_full[1]
        )
        # TODO: evaluate current at rho=0 nodes instead of extrapolation
        if not eq.sym:
            bsubsmnc[:, :] = c
            bsubsmnc[0, :] = (
                c[1, :] - (c[2, :] - c[1, :]) / (s_full[2] - s_full[1]) * s_full[1]
            )
        timer.stop("B_psi")
        if verbose > 1:
            timer.disp("B_psi")

        # B_theta
        timer.start("B_theta")
        if verbose > 0:
            print("Saving B_theta")
        bsubumnc = file.createVariable(
            "bsubumnc", np.float64, ("radius", "mn_mode_nyq")
        )
        bsubumnc.long_name = "cos(m*t-n*p) component of B_theta, on half mesh"
        bsubumnc.units = "T*m"
        m = cos_basis.modes[:, 1]
        n = cos_basis.modes[:, 2]
        if not eq.sym:
            bsubumns = file.createVariable(
                "bsubumns", np.float64, ("radius", "mn_mode_nyq")
            )
            bsubumns.long_name = "sin(m*t-n*p) component of B_theta, on half mesh"
            bsubumns.units = "T*m"
            m = full_basis.modes[:, 1]
            n = full_basis.modes[:, 2]
        x_mn = np.zeros((surfs - 1, m.size))
        for k in range(surfs - 1):
            grid = LinearGrid(M=2 * M_nyq + 1, N=2 * N_nyq + 1, NFP=NFP, rho=r_half[k])
            data = eq.compute_magnetic_field(grid)["B_theta"]
            if eq.sym:
                x_mn[k, :] = cos_transform.fit(data)
            else:
                x_mn[k, :] = full_transform.fit(data)
        xm, xn, s, c = ptolemy_identity_rev(m, n, x_mn)
        bsubumnc[0, :] = 0
        bsubumnc[1:, :] = c
        if not eq.sym:
            bsubumns[0, :] = 0
            bsubumns[1:, :] = s
        timer.stop("B_theta")
        if verbose > 1:
            timer.disp("B_theta")

        # B_zeta
        timer.start("B_zeta")
        if verbose > 0:
            print("Saving B_zeta")
        bsubvmnc = file.createVariable(
            "bsubvmnc", np.float64, ("radius", "mn_mode_nyq")
        )
        bsubvmnc.long_name = "cos(m*t-n*p) component of B_zeta, on half mesh"
        bsubvmnc.units = "T*m"
        m = cos_basis.modes[:, 1]
        n = cos_basis.modes[:, 2]
        if not eq.sym:
            bsubvmns = file.createVariable(
                "bsubvmns", np.float64, ("radius", "mn_mode_nyq")
            )
            bsubvmns.long_name = "sin(m*t-n*p) component of B_zeta, on half mesh"
            bsubvmns.units = "T*m"
            m = full_basis.modes[:, 1]
            n = full_basis.modes[:, 2]
        x_mn = np.zeros((surfs - 1, m.size))
        for k in range(surfs - 1):
            grid = LinearGrid(M=2 * M_nyq + 1, N=2 * N_nyq + 1, NFP=NFP, rho=r_half[k])
            data = eq.compute_magnetic_field(grid)["B_zeta"]
            if eq.sym:
                x_mn[k, :] = cos_transform.fit(data)
            else:
                x_mn[k, :] = full_transform.fit(data)
        xm, xn, s, c = ptolemy_identity_rev(m, n, x_mn)
        bsubvmnc[0, :] = 0
        bsubvmnc[1:, :] = c
        if not eq.sym:
            bsubvmns[0, :] = 0
            bsubvmns[1:, :] = s
        timer.stop("B_zeta")
        if verbose > 1:
            timer.disp("B_zeta")

        # J^theta * sqrt(g)
        timer.start("J^theta")
        if verbose > 0:
            print("Saving J^theta")
        currumnc = file.createVariable(
            "currumnc", np.float64, ("radius", "mn_mode_nyq")
        )
        currumnc.long_name = "cos(m*t-n*p) component of sqrt(g)*J^theta, on full mesh"
        currumnc.units = "A/m^3"
        m = cos_basis.modes[:, 1]
        n = cos_basis.modes[:, 2]
        if not eq.sym:
            currumns = file.createVariable(
                "currumns", np.float64, ("radius", "mn_mode_nyq")
            )
            currumns.long_name = (
                "sin(m*t-n*p) component of sqrt(g)*J^theta, on full mesh"
            )
            currumns.units = "A/m^3"
            m = full_basis.modes[:, 1]
            n = full_basis.modes[:, 2]
        x_mn = np.zeros((surfs, m.size))
        for k in range(surfs):
            grid = LinearGrid(M=2 * M_nyq + 1, N=2 * N_nyq + 1, NFP=NFP, rho=r_full[k])
            data = (
                eq.compute_current_density(grid)["J^theta"]
                * eq.compute_jacobian(grid)["g"]
            )
            if eq.sym:
                x_mn[k, :] = cos_transform.fit(data)
            else:
                x_mn[k, :] = full_transform.fit(data)
        xm, xn, s, c = ptolemy_identity_rev(m, n, x_mn)
        currumnc[:, :] = c
        currumnc[0, :] = (  # linear extrapolation for coefficient at the magnetic axis
            s[1, :] - (c[2, :] - c[1, :]) / (s_full[2] - s_full[1]) * s_full[1]
        )
        # TODO: evaluate current at rho=0 nodes instead of extrapolation
        if not eq.sym:
            currumns[:, :] = s
            currumns[0, :] = (
                s[1, :] - (s[2, :] - s[1, :]) / (s_full[2] - s_full[1]) * s_full[1]
            )
        timer.stop("J^theta")
        if verbose > 1:
            timer.disp("J^theta")

        # J^zeta * sqrt(g)
        timer.start("J^zeta")
        if verbose > 0:
            print("Saving J^zeta")
        currvmnc = file.createVariable(
            "currvmnc", np.float64, ("radius", "mn_mode_nyq")
        )
        currvmnc.long_name = "cos(m*t-n*p) component of sqrt(g)*J^zeta, on full mesh"
        currvmnc.units = "A/m^3"
        m = cos_basis.modes[:, 1]
        n = cos_basis.modes[:, 2]
        if not eq.sym:
            currvmns = file.createVariable(
                "currvmns", np.float64, ("radius", "mn_mode_nyq")
            )
            currvmns.long_name = (
                "sin(m*t-n*p) component of sqrt(g)*J^zeta, on full mesh"
            )
            currvmns.units = "A/m^3"
            m = full_basis.modes[:, 1]
            n = full_basis.modes[:, 2]
        x_mn = np.zeros((surfs, m.size))
        for k in range(surfs):
            grid = LinearGrid(M=2 * M_nyq + 1, N=2 * N_nyq + 1, NFP=NFP, rho=r_full[k])
            data = (
                eq.compute_current_density(grid)["J^zeta"]
                * eq.compute_jacobian(grid)["g"]
            )
            if eq.sym:
                x_mn[k, :] = cos_transform.fit(data)
            else:
                x_mn[k, :] = full_transform.fit(data)
        xm, xn, s, c = ptolemy_identity_rev(m, n, x_mn)
        currvmnc[:, :] = c
        currvmnc[0, :] = (  # linear extrapolation for coefficient at the magnetic axis
            s[1, :] - (c[2, :] - c[1, :]) / (s_full[2] - s_full[1]) * s_full[1]
        )
        # TODO: evaluate current at rho=0 nodes instead of extrapolation
        if not eq.sym:
            currvmns[:, :] = s
            currumns[0, :] = (
                s[1, :] - (s[2, :] - s[1, :]) / (s_full[2] - s_full[1]) * s_full[1]
            )
        timer.stop("J^zeta")
        if verbose > 1:
            timer.disp("J^zeta")

        file.close
        timer.stop("Total time")
        if verbose > 1:
            timer.disp("Total time")

    @classmethod
    def read_vmec_output(cls, fname):
        """Read VMEC data from wout NetCDF file.

        Parameters
        ----------
        fname : str or path-like
            filename of VMEC output file

        Returns
        -------
        vmec_data : dict
            the VMEC data fields

        """
        file = Dataset(fname, mode="r")

        vmec_data = {
            "NFP": file.variables["nfp"][:],
            "psi": file.variables["phi"][:],  # toroidal flux is saved as 'phi'
            "xm": file.variables["xm"][:],
            "xn": file.variables["xn"][:],
            "rmnc": file.variables["rmnc"][:],
            "zmns": file.variables["zmns"][:],
            "lmns": file.variables["lmns"][:],
        }
        try:
            vmec_data["rmns"] = file.variables["rmns"][:]
            vmec_data["zmnc"] = file.variables["zmnc"][:]
            vmec_data["lmnc"] = file.variables["lmnc"][:]
            vmec_data["sym"] = False
        except:
            vmec_data["sym"] = True

        return vmec_data

    @staticmethod
    def vmec_interpolate(Cmn, Smn, xm, xn, theta, phi, s=None, si=None, sym=True):
        """Interpolate VMEC data on a flux surface.

        Parameters
        ----------
        Cmn : ndarray
            cos(mt-np) Fourier coefficients
        Smn : ndarray
            sin(mt-np) Fourier coefficients
        xm : ndarray
            poloidal mode numbers
        xn : ndarray
            toroidal mode numbers
        theta : ndarray
            poloidal angles
        phi : ndarray
            toroidal angles
        s : ndarray
            radial coordinate, equivalent to normalized toroidal magnetic flux.
            Defaults to si (all flux surfaces)
        si : ndarray
            values of radial coordinates where Cmn,Smn are defined. Defaults to linearly
            spaced on [0,1]
        sym : bool
            stellarator symmetry (Default value = True)

        Returns
        -------
        if sym = True
            C, S (tuple of ndarray): VMEC data interpolated at the points (s,theta,phi)
            where C has cosine symmetry and S has sine symmetry
        if sym = False
            X (ndarray): non-symmetric VMEC data interpolated at the points (s,theta,phi)

        """
        if si is None:
            si = np.linspace(0, 1, Cmn.shape[0])
        if s is None:
            s = si
        Cf = interpolate.CubicSpline(si, Cmn)
        Sf = interpolate.CubicSpline(si, Smn)

        C = np.sum(
            Cf(s)
            * np.cos(
                xm[np.newaxis] * theta[:, np.newaxis]
                - xn[np.newaxis] * phi[:, np.newaxis]
            ),
            axis=-1,
        )
        S = np.sum(
            Sf(s)
            * np.sin(
                xm[np.newaxis] * theta[:, np.newaxis]
                - xn[np.newaxis] * phi[:, np.newaxis]
            ),
            axis=-1,
        )

        if sym:
            return C, S
        else:
            return C + S

    @classmethod
    def compute_theta_coords(cls, lmns, xm, xn, s, theta_star, zeta, si=None):
        """Find theta such that theta + lambda(theta) == theta_star.

        Parameters
        ----------
        lmns : array-like
            fourier coefficients for lambda
        xm : array-like
            poloidal mode numbers
        xn : array-like
            toroidal mode numbers
        s : array-like
            desired radial coordinates (normalized toroidal magnetic flux)
        theta_star : array-like
            desired straigh field-line poloidal angles (PEST/VMEC-like flux coordinates)
        zeta : array-like
            desired toroidal angles (toroidal coordinate phi)
        si : ndarray
            values of radial coordinates where lmns are defined. Defaults to linearly
            spaced on half grid between (0,1)

        Returns
        -------
        theta : ndarray
            theta such that theta + lambda(theta) == theta_star

        """
        if si is None:
            si = np.linspace(0, 1, lmns.shape[0])
            si[1:] = si[0:-1] + 0.5 / (lmns.shape[0] - 1)
        lmbda_mn = interpolate.CubicSpline(si, lmns)

        # Note: theta* (also known as vartheta) is the poloidal straight field-line
        # angle in PEST-like flux coordinates

        def root_fun(theta):
            lmbda = np.sum(
                lmbda_mn(s)
                * np.sin(
                    xm[np.newaxis] * theta[:, np.newaxis]
                    - xn[np.newaxis] * zeta[:, np.newaxis]
                ),
                axis=-1,
            )
            theta_star_k = theta + lmbda  # theta* = theta + lambda
            err = theta_star - theta_star_k
            return err

        out = optimize.root(
            root_fun, x0=theta_star, method="diagbroyden", options={"ftol": 1e-6}
        )
        return out.x

    @classmethod
    def area_difference_vmec(cls, equil, vmec_data, Nr=10, Nt=8, **kwargs):
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

        Returns
        -------
        area : float
            the average normalized area difference between flux surfaces area between
            flux surfaces is defined as the symmetric difference between the two shapes,
            and each is normalized to the nominal area of the flux surface, and finally
            averaged over the total number of flux surfaces being compared

        """
        # 1e-3 tolerance seems reasonable for testing, similar to comparison by eye
        if isinstance(vmec_data, (str, os.PathLike)):
            vmec_data = cls.read_vmec_output(vmec_data)

        if equil.N == 0:
            Nz = 1
        else:
            Nz = 6

        coords = cls.compute_coord_surfaces(equil, vmec_data, Nr, Nt, **kwargs)

        desc_poly = [
            [
                Polygon(np.array([R, Z]).T)
                for R, Z in zip(
                    coords["Rr_desc"][:, :, i].T, coords["Zr_desc"][:, :, i].T
                )
            ]
            for i in range(Nz)
        ]
        vmec_poly = [
            [
                Polygon(np.array([R, Z]).T)
                for R, Z in zip(
                    coords["Rr_vmec"][:, :, i].T, coords["Zr_vmec"][:, :, i].T
                )
            ]
            for i in range(Nz)
        ]

        return np.sum(
            [
                desc_poly[iz][ir].symmetric_difference(vmec_poly[iz][ir]).area
                / vmec_poly[iz][ir].area
                for ir in range(1, Nr)
                for iz in range(Nz)
            ]
        ) / ((Nr - 1) * Nz)

    @classmethod
    def compute_coord_surfaces(cls, equil, vmec_data, Nr=10, Nt=8, **kwargs):
        """Compute points on surfaces of constant rho, vartheta for both DESC and VMEC

        Parameters
        ----------
        equil : Equilibrium
            desc equilibrium to compare
        vmec_data : str or path-like or dict
            path to VMEC output file, or dictionary of vmec outputs
        Nr : int, optional
            number of rho contours
        Nt : int, optional
            number of vartheta contours

        Returns
        -------
        coords : dict of ndarray
            dictionary of coordinate arrays with keys Xy_code where X is R or Z, y is r
            for rho contours, or v for vartheta contours, and code is vmec or desc

        """
        if isinstance(vmec_data, (str, os.PathLike)):
            vmec_data = cls.read_vmec_output(vmec_data)

        if equil.N == 0:
            Nz = 1
        else:
            Nz = 6

        num_theta = kwargs.get("num_theta", 180)
        Nr_vmec = vmec_data["rmnc"].shape[0] - 1
        s_idx = Nr_vmec % np.floor(Nr_vmec / (Nr - 1))
        idxes = np.linspace(s_idx, Nr_vmec, Nr).astype(int)
        if s_idx != 0:
            idxes = np.pad(idxes, (1, 0), mode="constant")

        # flux surfaces to plot
        rr = np.sqrt(idxes / Nr_vmec)
        rt = np.linspace(0, 2 * np.pi, num_theta)
        rz = np.linspace(0, 2 * np.pi / equil.NFP, Nz, endpoint=False)
        r_grid = LinearGrid(rho=rr, theta=rt, zeta=rz)

        # straight field-line angles to plot
        tr = np.linspace(0, 1, 50)
        tt = np.linspace(0, 2 * np.pi, Nt, endpoint=False)
        tz = np.linspace(0, 2 * np.pi / equil.NFP, Nz, endpoint=False)
        t_grid = LinearGrid(rho=tr, theta=tt, zeta=tz)

        # Note: theta* (also known as vartheta) is the poloidal straight field-line
        # angle in PEST-like flux coordinates

        # find theta angles corresponding to desired theta* angles
        v_grid = Grid(equil.compute_theta_coords(t_grid.nodes))
        r_coords_desc = equil.compute_toroidal_coords(r_grid)
        v_coords_desc = equil.compute_toroidal_coords(v_grid)

        # rho contours
        Rr_desc = r_coords_desc["R"].reshape((r_grid.M, r_grid.L, r_grid.N), order="F")
        Zr_desc = r_coords_desc["Z"].reshape((r_grid.M, r_grid.L, r_grid.N), order="F")

        # vartheta contours
        Rv_desc = v_coords_desc["R"].reshape((t_grid.M, t_grid.L, t_grid.N), order="F")
        Zv_desc = v_coords_desc["Z"].reshape((t_grid.M, t_grid.L, t_grid.N), order="F")

        # Note: the VMEC radial coordinate s is the normalized toroidal magnetic flux;
        # the DESC radial coordiante rho = sqrt(s)

        # convert from rho -> s
        r_nodes = r_grid.nodes
        r_nodes[:, 0] = r_nodes[:, 0] ** 2
        t_nodes = t_grid.nodes
        t_nodes[:, 0] = t_nodes[:, 0] ** 2

        v_nodes = cls.compute_theta_coords(
            vmec_data["lmns"],
            vmec_data["xm"],
            vmec_data["xn"],
            t_nodes[:, 0],
            t_nodes[:, 1],
            t_nodes[:, 2],
        )

        t_nodes[:, 1] = v_nodes

        Rr_vmec, Zr_vmec = cls.vmec_interpolate(
            vmec_data["rmnc"],
            vmec_data["zmns"],
            vmec_data["xm"],
            vmec_data["xn"],
            theta=r_nodes[:, 1],
            phi=r_nodes[:, 2],
            s=r_nodes[:, 0],
        )

        Rv_vmec, Zv_vmec = cls.vmec_interpolate(
            vmec_data["rmnc"],
            vmec_data["zmns"],
            vmec_data["xm"],
            vmec_data["xn"],
            theta=t_nodes[:, 1],
            phi=t_nodes[:, 2],
            s=t_nodes[:, 0],
        )

        coords = {
            "Rr_desc": Rr_desc,
            "Zr_desc": Zr_desc,
            "Rv_desc": Rv_desc,
            "Zv_desc": Zv_desc,
            "Rr_vmec": Rr_vmec.reshape((r_grid.M, r_grid.L, r_grid.N), order="F"),
            "Zr_vmec": Zr_vmec.reshape((r_grid.M, r_grid.L, r_grid.N), order="F"),
            "Rv_vmec": Rv_vmec.reshape((t_grid.M, t_grid.L, t_grid.N), order="F"),
            "Zv_vmec": Zv_vmec.reshape((t_grid.M, t_grid.L, t_grid.N), order="F"),
        }
        return coords

    @classmethod
    def plot_vmec_comparison(cls, equil, vmec_data, Nr=10, Nt=8, **kwargs):
        """Plot a comparison to VMEC flux surfaces.

        Parameters
        ----------
        equil : Equilibrium
            desc equilibrium to compare
        vmec_data : str or path-like or dict
            path to VMEC output file, or dictionary of vmec outputs
        Nr : int, optional
            number of rho contours to plot
        Nt : int, optional
            number of vartheta contours to plot

        Returns
        -------
        fig : matplotlib.figure.Figure
            figure being plotted to
        ax : matplotlib.axes.Axes or ndarray of Axes
            axes being plotted to

        """
        if isinstance(vmec_data, (str, os.PathLike)):
            vmec_data = cls.read_vmec_output(vmec_data)
        coords = cls.compute_coord_surfaces(equil, vmec_data, Nr, Nt, **kwargs)

        if equil.N == 0:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6), squeeze=False)
        else:
            fig, ax = plt.subplots(2, 3, figsize=(16, 12), squeeze=False)
        ax = ax.flatten()

        for k in range(len(ax)):
            ax[k].plot(coords["Rr_vmec"][0, 0, k], coords["Zr_vmec"][0, 0, k], "bo")
            s_vmec = ax[k].plot(
                coords["Rr_vmec"][:, :, k], coords["Zr_vmec"][:, :, k], "b-"
            )
            ax[k].plot(coords["Rv_vmec"][:, :, k].T, coords["Zv_vmec"][:, :, k].T, "b-")

            ax[k].plot(coords["Rr_desc"][0, 0, k], coords["Zr_desc"][0, 0, k], "ro")
            ax[k].plot(coords["Rv_desc"][:, :, k].T, coords["Zv_desc"][:, :, k].T, "r:")
            s_desc = ax[k].plot(
                coords["Rr_desc"][:, :, k], coords["Zr_desc"][:, :, k], "r:"
            )

            ax[k].axis("equal")
            ax[k].set_xlabel(r"$R ~(\mathrm{m})$")
            ax[k].set_ylabel(r"$Z ~(\mathrm{m})$")
            if k == 0:
                s_vmec[0].set_label(r"$\mathrm{VMEC}$")
                s_desc[0].set_label(r"$\mathrm{DESC}$")
                ax[k].legend(fontsize="x-small")

        return fig, ax
