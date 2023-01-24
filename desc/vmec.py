"""Functions and classes for interfacing with VMEC equilibria."""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset, stringtochar
from scipy import integrate, interpolate, optimize

from desc.backend import sign
from desc.basis import DoubleFourierSeries
from desc.compat import ensure_positive_jacobian
from desc.compute.utils import compress
from desc.equilibrium import Equilibrium
from desc.grid import Grid, LinearGrid
from desc.objectives import FixBoundaryR, FixBoundaryZ, ObjectiveFunction
from desc.objectives.utils import factorize_linear_constraints
from desc.profiles import PowerSeriesProfile
from desc.transform import Transform
from desc.utils import Timer
from desc.vmec_utils import (
    fourier_to_zernike,
    ptolemy_identity_fwd,
    ptolemy_identity_rev,
    zernike_to_fourier,
)


class VMECIO:
    """Performs input from VMEC netCDF files to DESC Equilibrium and vice-versa."""

    @classmethod
    def load(
        cls, path, L=None, M=None, N=None, spectral_indexing="ansi", profile="iota"
    ):
        """Load a VMEC netCDF file as an Equilibrium.

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
        profile : {"iota", "current"}
             Which profile to use as the equilibrium constraint. (Default = ``'iota'``)

        Returns
        -------
        eq: Equilibrium
            Equilibrium that resembles the VMEC data.

        """
        assert profile in ["iota", "current"]
        file = Dataset(path, mode="r")
        inputs = {}

        version = file.variables["version_"][0]
        if version < 9:
            warnings.warn(
                "VMEC output appears to be from version {}".format(str(version))
                + " while DESC is only designed for compatibility with VMEC version"
                + " 9. Some data may not be loaded correctly."
            )

        # parameters
        inputs["Psi"] = float(file.variables["phi"][-1])
        inputs["NFP"] = int(file.variables["nfp"][0])
        inputs["M"] = M if M is not None else int(file.variables["mpol"][0] - 1)
        inputs["N"] = N if N is not None else int(file.variables["ntor"][0])
        inputs["spectral_indexing"] = spectral_indexing
        default_L = {
            "ansi": inputs["M"],
            "fringe": 2 * inputs["M"],
        }
        inputs["L"] = L if L is not None else default_L[inputs["spectral_indexing"]]

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
        except KeyError:
            rmns = np.zeros_like(rmnc)
            zmnc = np.zeros_like(zmns)
            lmnc = np.zeros_like(lmns)
            inputs["sym"] = True

        # profiles
        preset = file.dimensions["preset"].size
        pmass_type = "".join(file.variables["pmass_type"][:].astype(str)).strip()
        if pmass_type != "power_series":
            warnings.warn("Pressure is not a power series!")
        p0 = file.variables["presf"][0] / file.variables["am"][0]
        inputs["pressure"] = np.zeros((preset, 2))
        inputs["pressure"][:, 0] = np.arange(0, 2 * preset, 2)
        inputs["pressure"][:, 1] = file.variables["am"][:] * p0
        if profile == "iota":
            piota_type = "".join(file.variables["piota_type"][:].astype(str)).strip()
            if piota_type != "power_series":
                warnings.warn("Iota is not a power series!")
            inputs["iota"] = np.zeros((preset, 2))
            inputs["iota"][:, 0] = np.arange(0, 2 * preset, 2)
            inputs["iota"][:, 1] = file.variables["ai"][:]
            inputs["current"] = None
        if profile == "current":
            pcurr_type = "".join(file.variables["pcurr_type"][:].astype(str)).strip()
            if pcurr_type != "power_series":
                warnings.warn("Current is not a power series!")
            inputs["current"] = np.zeros((preset, 2))
            inputs["current"][:, 0] = np.arange(0, 2 * preset, 2)
            inputs["current"][:, 1] = file.variables["ac"][:]
            # integrate current profile wrt s=rho^2
            inputs["current"] = np.pad(
                np.vstack(
                    (
                        inputs["current"][:, 0] + 2,
                        inputs["current"][:, 1] * 2 / (inputs["current"][:, 0] + 2),
                    )
                ).T,
                ((1, 0), (0, 0)),
            )
            # scale total current to correct value
            inputs["current"][:, 1] *= (
                file.variables["ctor"][:] / (np.sum(inputs["current"][:, 1]) or 1),
            )
            inputs["iota"] = None

        file.close()

        # boundary
        m, n, Rb_lmn = ptolemy_identity_fwd(xm, xn, s=rmns[-1, :], c=rmnc[-1, :])
        m, n, Zb_lmn = ptolemy_identity_fwd(xm, xn, s=zmns[-1, :], c=zmnc[-1, :])
        inputs["surface"] = np.vstack((np.zeros_like(m), m, n, Rb_lmn, Zb_lmn)).T

        # initialize Equilibrium
        eq = Equilibrium(**inputs)

        # R
        m, n, R_mn = ptolemy_identity_fwd(xm, xn, s=rmns, c=rmnc)
        eq.R_lmn = fourier_to_zernike(m, n, R_mn, eq.R_basis)

        # Z
        m, n, Z_mn = ptolemy_identity_fwd(xm, xn, s=zmns, c=zmnc)
        eq.Z_lmn = fourier_to_zernike(m, n, Z_mn, eq.Z_basis)

        # lambda
        m, n, L_mn = ptolemy_identity_fwd(xm, xn, s=lmns, c=lmnc)
        eq.L_lmn = fourier_to_zernike(m, n, L_mn, eq.L_basis)

        eq = ensure_positive_jacobian(eq)

        # apply boundary conditions
        constraints = (
            FixBoundaryR(fixed_boundary=True),
            FixBoundaryZ(fixed_boundary=True),
        )
        objective = ObjectiveFunction(constraints, eq=eq, verbose=0)
        _, _, _, _, _, _, project, recover = factorize_linear_constraints(
            constraints, objective.args
        )
        args = objective.unpack_state(recover(project(objective.x(eq))))
        for key, value in args.items():
            setattr(eq, key, value)

        return eq

    @classmethod
    def save(cls, eq, path, surfs=128, verbose=1):  # noqa: C901 - FIXME - simplify
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
        signgs[:] = -sign(
            eq.compute("sqrt(g)", grid=Grid(np.array([[1, 0, 0]])))["sqrt(g)"]
        )

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
            s_full, eq.pressure(r_full), order=10, sym=False
        ).params

        ai = file.createVariable("ai", np.float64, ("preset",))
        ai.long_name = "rotational transform coefficients"
        ai[:] = np.zeros((file.dimensions["preset"].size,))
        if eq.iota is not None:
            # only using up to 10th order to avoid poor conditioning
            ai[:11] = (
                signgs[:]
                * PowerSeriesProfile.from_values(
                    s_full, eq.iota(r_full), order=10, sym=False
                ).params
            )

        ac = file.createVariable("ac", np.float64, ("preset",))
        ac.long_name = "normalized toroidal current density coefficients"
        ac[:] = np.zeros((file.dimensions["preset"].size,))
        if eq.current is not None:
            # only using up to 10th order to avoid poor conditioning
            ac[:11] = PowerSeriesProfile.from_values(
                s_full, eq.current(r_full), order=10, sym=False
            ).params

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
        if eq.iota is not None:
            iotaf[:] = signgs[:] * eq.iota(r_full)
        else:
            # value closest to axis will be nan
            grid = LinearGrid(M=12, N=12, rho=r_full, NFP=NFP)
            iotaf[:] = signgs[:] * compress(grid, eq.compute("iota", grid=grid)["iota"])

        iotas = file.createVariable("iotas", np.float64, ("radius",))
        iotas.long_name = "rotational transform on half mesh"
        iotas[0] = 0
        if eq.iota is not None:
            iotas[1:] = signgs[:] * eq.iota(r_half)
        else:
            grid = LinearGrid(M=12, N=12, rho=r_half, NFP=NFP)
            iotas[1:] = signgs[:] * compress(
                grid, eq.compute("iota", grid=grid)["iota"]
            )

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
        chi[:] = (
            2
            * Psi
            * signgs[:]
            * integrate.cumtrapz(r_full * iotaf[:], r_full, initial=0)
        )

        chipf = file.createVariable("chipf", np.float64, ("radius",))
        chipf.long_name = "d(chi)/ds: poloidal flux derivative"
        chipf[:] = phipf[:] * iotaf[:]

        Rmajor_p = file.createVariable("Rmajor_p", np.float64)
        Rmajor_p.long_name = "major radius"
        Rmajor_p.units = "m"
        Rmajor_p[:] = eq.compute("R0")["R0"]

        Aminor_p = file.createVariable("Aminor_p", np.float64)
        Aminor_p.long_name = "minor radius"
        Aminor_p.units = "m"
        Aminor_p[:] = eq.compute("a")["a"]

        aspect = file.createVariable("aspect", np.float64)
        aspect.long_name = "aspect ratio = R_major / A_minor"
        aspect[:] = eq.compute("R0/a")["R0/a"]

        volume_p = file.createVariable("volume_p", np.float64)
        volume_p.long_name = "plasma volume"
        volume_p.units = "m^3"
        volume_p[:] = eq.compute("V")["V"]

        grid = LinearGrid(M=eq.M_grid, N=eq.M_grid, NFP=eq.NFP, sym=eq.sym, rho=r_full)
        data = eq.compute(["D_Mercier", "I", "G"], grid=grid)

        # Boozer currents
        buco = file.createVariable("buco", np.float64, ("radius",))
        buco.long_name = "Boozer toroidal current I"
        buco.units = "T*m"
        buco[:] = compress(grid, data["I"])
        buco[0] = 0

        bvco = file.createVariable("bvco", np.float64, ("radius",))
        bvco.long_name = "Boozer poloidal current G"
        bvco.units = "T*m"
        bvco[:] = compress(grid, data["G"])
        bvco[0] = 0

        # Mercier stability
        DShear = file.createVariable("DShear", np.float64, ("radius",))
        DShear.long_name = "Mercier stability criterion magnetic shear term"
        DShear[:] = compress(grid, data["D_shear"])

        DCurr = file.createVariable("DCurr", np.float64, ("radius",))
        DCurr.long_name = "Mercier stability criterion toroidal current term"
        DCurr[:] = compress(grid, data["D_current"])

        DWell = file.createVariable("DWell", np.float64, ("radius",))
        DWell.long_name = "Mercier stability criterion magnetic well term"
        DWell[:] = compress(grid, data["D_well"])

        DGeod = file.createVariable("DGeod", np.float64, ("radius",))
        DGeod.long_name = "Mercier stability criterion geodesic curvature term"
        DGeod[:] = compress(grid, data["D_geodesic"])

        DMerc = file.createVariable("DMerc", np.float64, ("radius",))
        DMerc.long_name = "Mercier stability criterion"
        DMerc[:] = compress(grid, data["D_Mercier"])

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
            zaxis_cc[:] = Z0_n[N:]

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
        r1 = np.ones_like(eq.R_lmn)
        r1[eq.R_basis.modes[:, 1] < 0] *= -1
        m, n, x_mn = zernike_to_fourier(r1 * eq.R_lmn, basis=eq.R_basis, rho=r_full)
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
        z1 = np.ones_like(eq.Z_lmn)
        z1[eq.Z_basis.modes[:, 1] < 0] *= -1
        m, n, x_mn = zernike_to_fourier(z1 * eq.Z_lmn, basis=eq.Z_basis, rho=r_full)
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
        l1 = np.ones_like(eq.L_lmn)
        l1[eq.L_basis.modes[:, 1] >= 0] *= -1
        m, n, x_mn = zernike_to_fourier(l1 * eq.L_lmn, basis=eq.L_basis, rho=r_half)
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

        grid = LinearGrid(M=M_nyq, N=N_nyq, NFP=NFP)
        coords = eq.compute(["R", "Z"], grid=grid)
        sin_basis = DoubleFourierSeries(M=M_nyq, N=N_nyq, NFP=NFP, sym="sin")
        cos_basis = DoubleFourierSeries(M=M_nyq, N=N_nyq, NFP=NFP, sym="cos")
        full_basis = DoubleFourierSeries(M=M_nyq, N=N_nyq, NFP=NFP, sym=None)
        if eq.sym:
            sin_transform = Transform(
                grid=grid, basis=sin_basis, build=False, build_pinv=True
            )
            cos_transform = Transform(
                grid=grid, basis=cos_basis, build=False, build_pinv=True
            )
        else:
            full_transform = Transform(
                grid=grid, basis=full_basis, build=False, build_pinv=True
            )

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

        # half grid quantities
        half_grid = LinearGrid(M=M_nyq, N=N_nyq, NFP=NFP, rho=r_half)
        data_half_grid = eq.compute(
            ["J", "|B|", "B_rho", "B_theta", "B_zeta"], grid=half_grid
        )

        # full grid quantities
        full_grid = LinearGrid(M=M_nyq, N=N_nyq, NFP=NFP, rho=r_full)
        data_full_grid = eq.compute(["J", "B_rho", "B_theta", "B_zeta"], grid=full_grid)

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
        # d(rho)/d(s) = 1/(2*rho)
        data = (
            (data_half_grid["sqrt(g)"] / (2 * data_half_grid["rho"]))
            .reshape(
                (half_grid.num_theta, half_grid.num_rho, half_grid.num_zeta), order="F"
            )
            .transpose((1, 0, 2))
            .reshape((half_grid.num_rho, -1), order="F")
        )
        x_mn = np.zeros((surfs - 1, m.size))
        for i in range(surfs - 1):
            if eq.sym:
                x_mn[i, :] = cos_transform.fit(data[i, :])
            else:
                x_mn[i, :] = full_transform.fit(data[i, :])
        xm, xn, s, c = ptolemy_identity_rev(m, n, x_mn)
        gmnc[0, :] = 0
        gmnc[1:, :] = c * signgs[:]
        if not eq.sym:
            gmns[0, :] = 0
            gmns[1:, :] = s * signgs[:]
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
        data = (
            data_half_grid["|B|"]
            .reshape(
                (half_grid.num_theta, half_grid.num_rho, half_grid.num_zeta), order="F"
            )
            .transpose((1, 0, 2))
            .reshape((half_grid.num_rho, -1), order="F")
        )
        x_mn = np.zeros((surfs - 1, m.size))
        for i in range(surfs - 1):
            if eq.sym:
                x_mn[i, :] = cos_transform.fit(data[i, :])
            else:
                x_mn[i, :] = full_transform.fit(data[i, :])
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
        data = (
            data_half_grid["B^theta"]
            .reshape(
                (half_grid.num_theta, half_grid.num_rho, half_grid.num_zeta), order="F"
            )
            .transpose((1, 0, 2))
            .reshape((half_grid.num_rho, -1), order="F")
        )
        x_mn = np.zeros((surfs - 1, m.size))
        for i in range(surfs - 1):
            if eq.sym:
                x_mn[i, :] = cos_transform.fit(data[i, :])
            else:
                x_mn[i, :] = full_transform.fit(data[i, :])
        xm, xn, s, c = ptolemy_identity_rev(m, n, x_mn)
        bsupumnc[0, :] = 0
        bsupumnc[1:, :] = c * signgs[:]
        if not eq.sym:
            bsupumns[0, :] = 0
            bsupumns[1:, :] = s * signgs[:]
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
        data = (
            data_half_grid["B^zeta"]
            .reshape(
                (half_grid.num_theta, half_grid.num_rho, half_grid.num_zeta), order="F"
            )
            .transpose((1, 0, 2))
            .reshape((half_grid.num_rho, -1), order="F")
        )
        x_mn = np.zeros((surfs - 1, m.size))
        for i in range(surfs - 1):
            if eq.sym:
                x_mn[i, :] = cos_transform.fit(data[i, :])
            else:
                x_mn[i, :] = full_transform.fit(data[i, :])
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
        data = data_full_grid["B_rho"].reshape(
            (full_grid.num_theta, full_grid.num_rho, full_grid.num_zeta), order="F"
        ).transpose((1, 0, 2)).reshape((full_grid.num_rho, -1), order="F") / (
            2 * r_full[:, np.newaxis]
        )
        # B_rho -> B_psi conversion: d(rho)/d(s) = 1/(2*rho)
        x_mn = np.zeros((surfs, m.size))
        for i in range(surfs):
            if eq.sym:
                x_mn[i, :] = sin_transform.fit(data[i, :])
            else:
                x_mn[i, :] = full_transform.fit(data[i, :])
        xm, xn, s, c = ptolemy_identity_rev(m, n, x_mn)
        bsubsmns[:, :] = signgs[:] * s
        bsubsmns[0, :] = signgs[
            :
        ] * (  # linear extrapolation for coefficient at the magnetic axis
            s[1, :] - (s[2, :] - s[1, :]) / (s_full[2] - s_full[1]) * s_full[1]
        )
        # TODO: evaluate current at rho=0 nodes instead of extrapolation
        if not eq.sym:
            bsubsmnc[:, :] = signgs[:] * c
            bsubsmnc[0, :] = signgs[:] * (
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
        data = (
            data_half_grid["B_theta"]
            .reshape(
                (half_grid.num_theta, half_grid.num_rho, half_grid.num_zeta), order="F"
            )
            .transpose((1, 0, 2))
            .reshape((half_grid.num_rho, -1), order="F")
        )
        x_mn = np.zeros((surfs - 1, m.size))
        for i in range(surfs - 1):
            if eq.sym:
                x_mn[i, :] = cos_transform.fit(data[i, :])
            else:
                x_mn[i, :] = full_transform.fit(data[i, :])
        xm, xn, s, c = ptolemy_identity_rev(m, n, x_mn)
        bsubumnc[0, :] = 0
        bsubumnc[1:, :] = c * signgs[:]
        if not eq.sym:
            bsubumns[0, :] = 0
            bsubumns[1:, :] = s * signgs[:]
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
        data = (
            data_half_grid["B_zeta"]
            .reshape(
                (half_grid.num_theta, half_grid.num_rho, half_grid.num_zeta), order="F"
            )
            .transpose((1, 0, 2))
            .reshape((half_grid.num_rho, -1), order="F")
        )
        x_mn = np.zeros((surfs - 1, m.size))
        for i in range(surfs - 1):
            if eq.sym:
                x_mn[i, :] = cos_transform.fit(data[i, :])
            else:
                x_mn[i, :] = full_transform.fit(data[i, :])
        xm, xn, s, c = ptolemy_identity_rev(m, n, x_mn)
        bsubvmnc[0, :] = 0
        bsubvmnc[1:, :] = c
        if not eq.sym:
            bsubvmns[0, :] = 0
            bsubvmns[1:, :] = s
        timer.stop("B_zeta")
        if verbose > 1:
            timer.disp("B_zeta")

        # J^theta * sqrt(g)   # noqa: E800
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
        data = (
            (
                data_full_grid["J^theta"]
                * data_full_grid["sqrt(g)"]
                / (2 * data_full_grid["rho"])
            )
            .reshape(
                (full_grid.num_theta, full_grid.num_rho, full_grid.num_zeta), order="F"
            )
            .transpose((1, 0, 2))
            .reshape((full_grid.num_rho, -1), order="F")
        )
        x_mn = np.zeros((surfs, m.size))
        for i in range(surfs):
            if eq.sym:
                x_mn[i, :] = cos_transform.fit(data[i, :])
            else:
                x_mn[i, :] = full_transform.fit(data[i, :])
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

        # J^zeta * sqrt(g)   # noqa: E800
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
        data = (
            (
                data_full_grid["J^zeta"]
                * data_full_grid["sqrt(g)"]
                / (2 * data_full_grid["rho"])
            )
            .reshape(
                (full_grid.num_theta, full_grid.num_rho, full_grid.num_zeta), order="F"
            )
            .transpose((1, 0, 2))
            .reshape((full_grid.num_rho, -1), order="F")
        )
        x_mn = np.zeros((surfs, m.size))
        for i in range(surfs):
            if eq.sym:
                x_mn[i, :] = cos_transform.fit(data[i, :])
            else:
                x_mn[i, :] = full_transform.fit(data[i, :])
        xm, xn, s, c = ptolemy_identity_rev(m, n, x_mn)
        currvmnc[:, :] = signgs[:] * c
        currvmnc[0, :] = signgs[
            :
        ] * (  # linear extrapolation for coefficient at the magnetic axis
            s[1, :] - (c[2, :] - c[1, :]) / (s_full[2] - s_full[1]) * s_full[1]
        )
        # TODO: evaluate current at rho=0 nodes instead of extrapolation
        if not eq.sym:
            currvmns[:, :] = signgs[:] * s
            currumns[0, :] = signgs[:] * (
                s[1, :] - (s[2, :] - s[1, :]) / (s_full[2] - s_full[1]) * s_full[1]
            )
        timer.stop("J^zeta")
        if verbose > 1:
            timer.disp("J^zeta")

        # TODO: these output quantities need to be added
        """
        IonLarmor = file.createVariable("IonLarmor", np.float64)
        IonLarmor[:] = 0.0

        ac_aux_f = file.createVariable("ac_aux_f", np.float64, ("ndfmax",))
        ac_aux_f[:] = np.ones((file.dimensions["ndfmax"].size,)) * np.nan

        ac_aux_s = file.createVariable("ac_aux_s", np.float64, ("ndfmax",))
        ac_aux_s[:] = -np.ones((file.dimensions["ndfmax"].size,))

        ai_aux_f = file.createVariable("ai_aux_f", np.float64, ("ndfmax",))
        ai_aux_f[:] = np.ones((file.dimensions["ndfmax"].size,)) * np.nan

        ai_aux_s = file.createVariable("ai_aux_s", np.float64, ("ndfmax",))
        ai_aux_s[:] = -np.ones((file.dimensions["ndfmax"].size,))

        am_aux_f = file.createVariable("am_aux_f", np.float64, ("ndfmax",))
        am_aux_f[:] = np.ones((file.dimensions["ndfmax"].size,)) * np.nan

        am_aux_s = file.createVariable("am_aux_s", np.float64, ("ndfmax",))
        am_aux_s[:] = -np.ones((file.dimensions["ndfmax"].size,))

        b0 = file.createVariable("b0", np.float64)
        b0[:] = 1.0

        bdotb = file.createVariable("bdotb", np.float64, ("radius",))
        bdotb[:] = np.zeros((file.dimensions["radius"].size,))

        bdotgradv = file.createVariable("bdotgradv", np.float64, ("radius",))
        bdotgradv[:] = np.zeros((file.dimensions["radius"].size,))

        beta_vol = file.createVariable("beta_vol", np.float64, ("radius",))
        beta_vol[:] = np.zeros((file.dimensions["radius"].size,))

        betapol = file.createVariable("betapol", np.float64)
        betapol[:] = 0.0

        betator = file.createVariable("betator", np.float64)
        betator[:] = 0.0

        betatotal = file.createVariable("betatotal", np.float64)
        betatotal[:] = 0.0

        betaxis = file.createVariable("betaxis", np.float64)
        betaxis[:] = 0.0

        ctor = file.createVariable("ctor", np.float64)
        ctor[:] = 0.0

        extcur = file.createVariable("extcur", np.float64)
        extcur[:] = 0.0

        fsql = file.createVariable("fsql", np.float64)
        fsql[:] = 1e-16

        fsqr = file.createVariable("fsqr", np.float64)
        fsqr[:] = 1e-16

        fsqt = file.createVariable("fsqt", np.float64)
        fsqt[:] = 1e-16

        fsqz = file.createVariable("fsqz", np.float64)
        fsqz[:] = 1e-16

        ftolv = file.createVariable("ftolv", np.float64)
        ftolv[:] = 1e-16

        itfsq = file.createVariable("itfsq", np.int32)
        itfsq[:] = 1

        jcuru = file.createVariable("jcuru", np.float64, ("radius",))
        jcuru[:] = np.zeros((file.dimensions["radius"].size,))

        jcurv = file.createVariable("jcurv", np.float64, ("radius",))
        jcurv[:] = np.zeros((file.dimensions["radius"].size,))

        jdotb = file.createVariable("jdotb", np.float64, ("radius",))
        jdotb[:] = np.zeros((file.dimensions["radius"].size,))

        nextcur = file.createVariable("nextcur", np.int32)
        nextcur[:] = 0

        niter = file.createVariable("niter", np.int32)
        niter[:] = 1

        over_r = file.createVariable("over_r", np.float64, ("radius",))
        over_r[:] = np.zeros((file.dimensions["radius"].size,))

        q_factor = file.createVariable("q_factor", np.float64, ("radius",))
        q_factor[:] = np.zeros((file.dimensions["radius"].size,))

        rbtor = file.createVariable("rbtor", np.float64)
        rbtor[:] = 0.0

        rbtor0 = file.createVariable("rbtor0", np.float64)
        rbtor0[:] = 0.0

        specw = file.createVariable("specw", np.float64, ("radius",))
        specw[:] = np.zeros((file.dimensions["radius"].size,))

        volavgB = file.createVariable("volavgB", np.float64)
        volavgB[:] = 0.0

        vp = file.createVariable("vp", np.float64, ("radius",))
        vp[:] = np.zeros((file.dimensions["radius"].size,))

        wb = file.createVariable("wb", np.float64)
        wb[:] = 0.0

        wdot = file.createVariable("wdot", np.float64, ("time",))
        wdot[:] = np.zeros((file.dimensions["time"].size,))

        wp = file.createVariable("wp", np.float64)
        wp[:] = 0.0
        """

        file.close()
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
            "signgs": file.variables["signgs"][:],
        }
        try:
            vmec_data["rmns"] = file.variables["rmns"][:]
            vmec_data["zmnc"] = file.variables["zmnc"][:]
            vmec_data["lmnc"] = file.variables["lmnc"][:]
            vmec_data["sym"] = False
        except KeyError:
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
            X (ndarray): non-symmetric VMEC data interpolated at (s,theta,phi)
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
    def compute_coord_surfaces(cls, equil, vmec_data, Nr=10, Nt=8, Nz=None, **kwargs):
        """Compute points on surfaces of constant rho, vartheta for both DESC and VMEC.

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
        Nz : int, optional
            Number of zeta planes to compare. If None, use 1 plane for axisymmetric
            cases or 6 for non-axisymmetric.

        Returns
        -------
        coords : dict of ndarray
            dictionary of coordinate arrays with keys Xy_code where X is R or Z, y is r
            for rho contours, or v for vartheta contours, and code is vmec or desc

        """
        if isinstance(vmec_data, (str, os.PathLike)):
            vmec_data = cls.read_vmec_output(vmec_data)

        if Nz is None and equil.N == 0:
            Nz = 1
        elif Nz is None:
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
        r_grid = LinearGrid(rho=rr, theta=rt, zeta=rz, NFP=equil.NFP)

        # straight field-line angles to plot
        tr = np.linspace(0, 1, 50)
        tt = np.linspace(0, 2 * np.pi, Nt, endpoint=False)
        tz = np.linspace(0, 2 * np.pi / equil.NFP, Nz, endpoint=False)
        t_grid = LinearGrid(rho=tr, theta=tt, zeta=tz, NFP=equil.NFP)

        # Note: theta* (also known as vartheta) is the poloidal straight field-line
        # angle in PEST-like flux coordinates

        # find theta angles corresponding to desired theta* angles
        v_grid = Grid(equil.compute_theta_coords(t_grid.nodes))
        r_coords_desc = equil.compute(["R", "Z"], grid=r_grid)
        v_coords_desc = equil.compute(["R", "Z"], grid=v_grid)

        # rho contours
        Rr_desc = r_coords_desc["R"].reshape(
            (r_grid.num_theta, r_grid.num_rho, r_grid.num_zeta), order="F"
        )
        Zr_desc = r_coords_desc["Z"].reshape(
            (r_grid.num_theta, r_grid.num_rho, r_grid.num_zeta), order="F"
        )

        # vartheta contours
        Rv_desc = v_coords_desc["R"].reshape(
            (t_grid.num_theta, t_grid.num_rho, t_grid.num_zeta), order="F"
        )
        Zv_desc = v_coords_desc["Z"].reshape(
            (t_grid.num_theta, t_grid.num_rho, t_grid.num_zeta), order="F"
        )

        # Note: the VMEC radial coordinate s is the normalized toroidal magnetic flux;
        # the DESC radial coordinate rho = sqrt(s)

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
            "Rr_vmec": Rr_vmec.reshape(
                (r_grid.num_theta, r_grid.num_rho, r_grid.num_zeta), order="F"
            ),
            "Zr_vmec": Zr_vmec.reshape(
                (r_grid.num_theta, r_grid.num_rho, r_grid.num_zeta), order="F"
            ),
            "Rv_vmec": Rv_vmec.reshape(
                (t_grid.num_theta, t_grid.num_rho, t_grid.num_zeta), order="F"
            ),
            "Zv_vmec": Zv_vmec.reshape(
                (t_grid.num_theta, t_grid.num_rho, t_grid.num_zeta), order="F"
            ),
        }
        coords = {key: np.swapaxes(val, 0, 1) for key, val in coords.items()}
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
                coords["Rr_vmec"][:, :, k].T, coords["Zr_vmec"][:, :, k].T, "b-"
            )
            ax[k].plot(coords["Rv_vmec"][:, :, k], coords["Zv_vmec"][:, :, k], "b-")

            ax[k].plot(coords["Rr_desc"][0, 0, k].T, coords["Zr_desc"][0, 0, k].T, "ro")
            ax[k].plot(coords["Rv_desc"][:, :, k], coords["Zv_desc"][:, :, k], "r--")
            s_desc = ax[k].plot(
                coords["Rr_desc"][:, :, k].T, coords["Zr_desc"][:, :, k].T, "r--"
            )

            ax[k].axis("equal")
            ax[k].set_xlabel(r"$R ~(\mathrm{m})$")
            ax[k].set_ylabel(r"$Z ~(\mathrm{m})$")
            if k == 0:
                s_vmec[0].set_label(r"$\mathrm{VMEC}$")
                s_desc[0].set_label(r"$\mathrm{DESC}$")
                ax[k].legend(fontsize="x-small")

        return fig, ax
