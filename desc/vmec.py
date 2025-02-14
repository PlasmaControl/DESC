"""Functions and classes for interfacing with VMEC equilibria."""

import io
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset, stringtochar
from scipy import integrate, interpolate, optimize
from scipy.constants import mu_0

from desc.basis import DoubleFourierSeries
from desc.compat import ensure_positive_jacobian
from desc.equilibrium import Equilibrium
from desc.geometry import FourierRZToroidalSurface
from desc.grid import Grid, LinearGrid
from desc.integrals import surface_averages
from desc.objectives import (
    ObjectiveFunction,
    get_fixed_axis_constraints,
    get_fixed_boundary_constraints,
    maybe_add_self_consistency,
)
from desc.objectives.utils import factorize_linear_constraints
from desc.profiles import PowerSeriesProfile, SplineProfile
from desc.transform import Transform
from desc.utils import Timer, warnif
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
        """Load a VMEC netCDF file as an Equilibrium by fitting with Fourier-Zernike.

        Loads in the VMEC netCDF file and loads the R, Z and Lambda Fourier
        coefficients, which from VMEC are given on each discrete flux surface
        in the VMEC solution. A Fourier-Zernike basis is then fit to these
        R, Z and Lambda Fourier coefficients to yield the DESC representation
        that most closely resembles the VMEC solution. Finally, the VMEC
        boundary is loaded and the DESC Equilibrium R,Z are constrained to
        match the given VMEC boundary.

        NOTE: This is only a fit, so the DESC Equilibrium returned is not
        expected to be in force balance. It is recommended to solve the
        Equilibrium once loaded before using the Equilibrium for any
        analysis.

        Parameters
        ----------
        path : str
            File path of input data.
        L : int, optional
            Radial resolution of the fit. Default determined by index.
        M : int, optional
            Poloidal resolution of the fit. Default = MPOL-1 from VMEC solution.
        N : int, optional
            Toroidal resolution of the fit. Default = NTOR from VMEC solution.
        spectral_indexing : str, optional
            Type of Zernike indexing scheme to use. (Default = ``'ansi'``)
        profile : {"iota", "current"}
             Which profile to use as the equilibrium constraint. (Default = ``'iota'``)

        Returns
        -------
        eq: Equilibrium
            Equilibrium fit that resembles the VMEC data.

        Notes
        -----
        To ensure compatibility with different profile representations in VMEC,
        the resulting equilibrium will always have SplineProfile types for all profiles

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
        r = np.sqrt(np.linspace(0, 1, file.variables["ns"][:].filled()))
        pres = file.variables["presf"][:].filled()
        inputs["pressure"] = SplineProfile(pres, r, name="pressure")
        if profile == "iota":
            iota = file.variables["iotaf"][:].filled()
            inputs["iota"] = SplineProfile(iota, r, name="iota")
            inputs["current"] = None
        if profile == "current":
            curr = 2 * np.pi / mu_0 * file.variables["buco"][:].filled()
            inputs["current"] = SplineProfile(curr, r, name="current")
            inputs["iota"] = None

        # boundary
        m, n, Rb_lmn = ptolemy_identity_fwd(xm, xn, s=rmns[-1, :], c=rmnc[-1, :])
        m, n, Zb_lmn = ptolemy_identity_fwd(xm, xn, s=zmns[-1, :], c=zmnc[-1, :])
        surface = np.vstack((np.zeros_like(m), m, n, Rb_lmn, Zb_lmn)).T
        # need to create surface object here so we can tell it not to flip the
        # orientation yet. If we did it here, it would mess up the self-consistency
        # stuff later
        inputs["surface"] = FourierRZToroidalSurface(
            surface[:, 3],
            surface[:, 4],
            surface[:, 1:3].astype(int),
            surface[:, 1:3].astype(int),
            inputs["NFP"],
            inputs["sym"],
            check_orientation=False,
        )

        # axis
        rax_cc = file.variables["raxis_cc"][:].filled()
        zax_cs = file.variables["zaxis_cs"][:].filled()
        try:
            rax_cs = file.variables["raxis_cs"][:].filled()
            zax_cc = file.variables["zaxis_cc"][:].filled()
        except KeyError:
            rax_cs = np.zeros_like(rax_cc)
            zax_cc = np.zeros_like(zax_cs)
        rax = np.concatenate([-rax_cs[1:][::-1], rax_cc])
        zax = np.concatenate([-zax_cs[1:][::-1], zax_cc])
        nax = len(rax_cc) - 1
        nax = np.arange(-nax, nax + 1)
        inputs["axis"] = np.vstack([nax, rax, zax]).T

        file.close()

        # initialize Equilibrium
        eq = Equilibrium(**inputs, check_orientation=False, ensure_nested=False)

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
        constraints = get_fixed_axis_constraints(
            profiles=False, eq=eq
        ) + get_fixed_boundary_constraints(eq=eq)
        constraints = maybe_add_self_consistency(eq, constraints)
        objective = ObjectiveFunction(constraints)
        objective.build(verbose=0)
        _, _, _, _, _, _, project, recover, *_ = factorize_linear_constraints(
            objective, objective
        )
        args = objective.unpack_state(recover(project(objective.x(eq))), False)[0]
        eq.params_dict = args

        # now we flip the orientation at the very end
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Left handed coordinates detected"
            )
            eq = ensure_positive_jacobian(eq)

        return eq

    @classmethod
    def save(  # noqa: C901
        cls,
        eq,
        path,
        surfs=128,
        *,
        M_nyq=None,
        N_nyq=None,
        M_grid=None,
        N_grid=None,
        verbose=1,
    ):
        """Save an Equilibrium as a netCDF file in the VMEC format.

        Parameters
        ----------
        eq : Equilibrium
            Equilibrium to save.
        path : str
            File path of output data.
        surfs: int
            Number of flux surfaces to interpolate at (Default = 128).
        M_nyq, N_nyq: int
            The max poloidal and toroidal mode numbers to use in the Nyquist spectrum
            that the derived quantities are Fourier fit with. Defaults to M+4 and N+2.
        M_grid, N_grid: int
            The max poloidal and toroidal resolution of the grid to evaluate quantities
            in real space. Related to the VMEC inputs NTHETA and NZETA.
            Defaults to eq.M_grid and eq.N_grid.
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
        M_nyq = M + 4 if M_nyq is None else M_nyq
        warnif(
            N_nyq is not None and int(N) == 0,
            UserWarning,
            "Passed in N_nyq but equilibrium is axisymmetric, setting N_nyq to zero.",
        )
        N_nyq = N + 2 if N_nyq is None else N_nyq
        N_nyq = 0 if int(N) == 0 else N_nyq
        M_grid = eq.M_grid if M_grid is None else M_grid
        N_grid = eq.N_grid if N_grid is None else N_grid
        warnif(
            M_grid < M_nyq,
            UserWarning,
            f"M_grid = {M_grid} < M_nyq = {M_nyq}, "
            + "increase M_grid for a more accurate Fourier fit.",
        )
        warnif(
            N_grid < N_nyq,
            UserWarning,
            f"N_grid = {N_grid} < N_nyq = {N_nyq}, "
            + "increase N_grid for a more accurate Fourier fit.",
        )

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
        file.createDimension("time", 100)  # used for fsq* & wdot
        file.createDimension("dim_00001", 1)
        file.createDimension("dim_00020", 20)
        file.createDimension("dim_00100", 100)
        file.createDimension("dim_00200", 200)

        # compute data
        timer.start("compute")
        if verbose > 0:
            print("Computing data")

        grid_axis = LinearGrid(M=M_grid, N=N_grid, rho=np.array([0.0]), NFP=NFP)
        grid_lcfs = LinearGrid(M=M_grid, N=N_grid, rho=np.array([1.0]), NFP=NFP)
        grid_half = LinearGrid(M=M_grid, N=N_grid, NFP=NFP, rho=r_half)
        grid_full = LinearGrid(M=M_grid, N=N_grid, NFP=NFP, rho=r_full)

        data_quad = eq.compute(
            [
                "R0/a",
                "V",
                "W_B",
                "W_p",
                "<|B|>_rms",
                "<beta>_vol",
                "<beta_pol>_vol",
                "<beta_tor>_vol",
            ]
        )
        data_axis = eq.compute(["G", "p", "R", "<|B|^2>", "<|B|>"], grid=grid_axis)
        data_lcfs = eq.compute(["G", "I", "R", "Z"], grid=grid_lcfs)
        data_half = eq.compute(
            [
                "B_rho",
                "B_theta",
                "B_zeta",
                "G",
                "I",
                "J",
                "iota",
                "p",
                "sqrt(g)",
                "V_r(r)",
                "|B|",
                "<|B|^2>",
            ],
            grid=grid_half,
        )
        data_full = eq.compute(
            [
                "B_rho",
                "B_theta",
                "B_zeta",
                "current",
                "D_Mercier",
                "iota",
                "J",
                "J^theta*sqrt(g)",
                "J^zeta",
                "p",
                "sqrt(g)",
                "<|B|^2>",
                "<J*B>",
            ],
            grid=grid_full,
        )

        timer.stop("compute")
        if verbose > 1:
            timer.disp("compute")

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

        # TODO(#1378): instead of hard-coding for fixed-boundary,
        # also allow for free-boundary?
        mgrid_mode = file.createVariable("mgrid_mode", "S1", ("dim_00001",))
        mgrid_mode[:] = stringtochar(
            np.array([""], "S" + str(file.dimensions["dim_00001"].size))
        )

        mgrid_file = file.createVariable("mgrid_file", "S1", ("dim_00200",))
        mgrid_file[:] = stringtochar(
            np.array(["none" + " " * 196], "S" + str(file.dimensions["dim_00200"].size))
        )

        ier_flag = file.createVariable("ier_flag", np.int32)
        ier_flag.long_name = (
            "error flag (DESC always outputs 0; "
            + "manually check for a good equilibrium solution)"
        )
        ier_flag[:] = 0

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
        signgs.long_name = "sign of coordinate system Jacobian"
        signgs[:] = -1  # VMEC always uses a negative Jacobian

        gamma = file.createVariable("gamma", np.float64)
        gamma.long_name = "compressibility index (0 = pressure prescribed)"
        gamma[:] = 0

        nextcur = file.createVariable("nextcur", np.int32)
        nextcur.long_name = "number of coils (external currents)"
        nextcur[:] = 0  # hard-coded assuming fixed-boundary solve

        # TODO(#183): add option for saving spline profiles
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

        # scalars computed on a Quadrature grid

        Rmajor_p = file.createVariable("Rmajor_p", np.float64)
        Rmajor_p.long_name = "major radius"
        Rmajor_p.units = "m"
        Rmajor_p[:] = data_quad["R0"]

        Aminor_p = file.createVariable("Aminor_p", np.float64)
        Aminor_p.long_name = "minor radius"
        Aminor_p.units = "m"
        Aminor_p[:] = data_quad["a"]

        aspect = file.createVariable("aspect", np.float64)
        aspect.long_name = "aspect ratio = R_major / A_minor"
        aspect.units = "None"
        aspect[:] = data_quad["R0/a"]

        volume_p = file.createVariable("volume_p", np.float64)
        volume_p.long_name = "plasma volume"
        volume_p.units = "m^3"
        volume_p[:] = data_quad["V"]

        volavgB = file.createVariable("volavgB", np.float64)
        volavgB.long_name = "volume average magnetic field, root mean square"
        volavgB.units = "T"
        volavgB[:] = data_quad["<|B|>_rms"]

        betatotal = file.createVariable("betatotal", np.float64)
        betatotal.long_name = "normalized plasma pressure"
        betatotal.units = "None"
        betatotal[:] = data_quad["<beta>_vol"]

        betapol = file.createVariable("betapol", np.float64)
        betapol.long_name = "normalized poloidal plasma pressure"
        betapol.units = "None"
        betapol[:] = data_quad["<beta_pol>_vol"]

        betator = file.createVariable("betator", np.float64)
        betator.long_name = "normalized toroidal plasma pressure"
        betator.units = "None"
        betator[:] = data_quad["<beta_tor>_vol"]

        wb = file.createVariable("wb", np.float64)
        wb.long_name = "plasma magnetic energy * mu_0/(4*pi^2)"
        wb.units = "T^2*m^3"
        wb[:] = data_quad["W_B"] * mu_0 / (4 * np.pi**2)

        wp = file.createVariable("wp", np.float64)
        wp.long_name = "plasma thermodynamic energy * mu_0/(4*pi^2)"
        wp.units = "T^2*m^3"
        wp[:] = np.abs(data_quad["W_p"]) * mu_0 / (4 * np.pi**2)

        # scalars computed at the magnetic axis

        rbtor0 = file.createVariable("rbtor0", np.float64)
        rbtor0.long_name = "<R*B_tor> on axis"
        rbtor0.units = "T*m"
        rbtor0[:] = data_axis["G"][0]

        b0 = file.createVariable("b0", np.float64)
        b0.long_name = "average B_tor on axis"
        b0.units = "T"
        b0[:] = data_axis["<|B|>"][0]

        betaxis = file.createVariable("betaxis", np.float64)
        betaxis.long_name = "2 * mu_0 * pressure / <|B|^2> on the magnetic axis"
        betaxis.units = "None"
        betaxis[:] = 2 * mu_0 * data_axis["p"][0] / data_axis["<|B|^2>"][0]

        # scalars computed at the last closed flux surface

        ctor = file.createVariable("ctor", np.float64)
        ctor.long_name = "total toroidal plasma current"
        ctor.units = "A"
        ctor[:] = data_lcfs["I"][0] * 2 * np.pi / mu_0

        rbtor = file.createVariable("rbtor", np.float64)
        rbtor.long_name = "<R*B_tor> on last closed flux surface"
        rbtor.units = "T*m"
        rbtor[:] = data_lcfs["G"][0]

        # half mesh quantities

        pres = file.createVariable("pres", np.float64, ("radius",))
        pres.long_name = "pressure on half mesh"
        pres.units = "Pa"
        pres[0] = 0
        pres[1:] = grid_half.compress(data_half["p"])

        mass = file.createVariable("mass", np.float64, ("radius",))
        mass.long_name = "mass on half mesh"
        mass.units = "Pa"
        mass[:] = pres[:]

        iotas = file.createVariable("iotas", np.float64, ("radius",))
        iotas.long_name = "rotational transform on half mesh"
        iotas.units = "None"
        iotas[0] = 0
        iotas[1:] = -grid_half.compress(data_half["iota"])  # - for negative Jacobian

        phips = file.createVariable("phips", np.float64, ("radius",))
        phips.long_name = "d(phi)/ds * -1/2pi: toroidal flux derivative, on half mesh"
        phips[0] = 0
        phips[1:] = -Psi * np.ones((surfs - 1,)) / (2 * np.pi)

        buco = file.createVariable("buco", np.float64, ("radius",))
        buco.long_name = "Boozer toroidal current I, on half mesh"
        buco.units = "T*m"
        buco[1:] = -grid_half.compress(data_half["I"])  # - for negative Jacobian
        buco[0] = 0

        bvco = file.createVariable("bvco", np.float64, ("radius",))
        bvco.long_name = "Boozer poloidal current G, on half mesh"
        bvco.units = "T*m"
        bvco[1:] = grid_half.compress(data_half["G"])
        bvco[0] = 0

        beta_vol = file.createVariable("beta_vol", np.float64, ("radius",))
        beta_vol.long_name = "2 * mu_0 * pressure / <|B|^2>, on half mesh"
        beta_vol.units = "None"
        beta_vol[1:] = (
            2
            * mu_0
            * grid_half.compress(data_half["p"])
            / grid_half.compress(data_half["<|B|^2>"])
        )
        beta_vol[0] = 0.0

        vp = file.createVariable("vp", np.float64, ("radius",))
        vp.long_name = "dV/ds normalized by 4*pi^2, on half mesh"
        vp.units = "m^3"
        vp[1:] = grid_half.compress(data_half["V_r(r)"]) / (
            8 * np.pi**2 * grid_half.compress(data_half["rho"])
        )
        vp[0] = 0

        # full mesh quantities

        presf = file.createVariable("presf", np.float64, ("radius",))
        presf.long_name = "pressure on full mesh"
        presf.units = "Pa"
        presf[:] = grid_full.compress(data_full["p"])

        iotaf = file.createVariable("iotaf", np.float64, ("radius",))
        iotaf.long_name = "rotational transform on full mesh"
        iotaf.units = "None"
        iotaf[:] = -grid_full.compress(data_full["iota"])  # - for negative Jacobian

        q_factor = file.createVariable("q_factor", np.float64, ("radius",))
        q_factor.long_name = "inverse rotational transform on full mesh"
        q_factor.units = "None"
        q_factor[:] = 1 / iotaf[:]

        phi = file.createVariable("phi", np.float64, ("radius",))
        phi.long_name = "toroidal flux, on full mesh"
        phi.units = "Wb"
        phi[:] = np.linspace(0, Psi, surfs)

        phipf = file.createVariable("phipf", np.float64, ("radius",))
        phipf.long_name = "d(phi)/ds: toroidal flux derivative, on full mesh"
        phipf[:] = Psi * np.ones((surfs,))

        chi = file.createVariable("chi", np.float64, ("radius",))
        chi.long_name = "poloidal flux, on full mesh"
        chi.units = "Wb"
        chi[:] = (
            -2  # - for negative Jacobian
            * Psi
            * integrate.cumulative_trapezoid(r_full * iotaf[:], r_full, initial=0)
        )

        chipf = file.createVariable("chipf", np.float64, ("radius",))
        chipf.long_name = "d(chi)/ds: poloidal flux derivative, on full mesh"
        chipf[:] = phipf[:] * iotaf[:]

        am = file.createVariable("am", np.float64, ("preset",))
        am.long_name = "pressure coefficients"
        am.units = "Pa"
        am[:] = np.zeros((file.dimensions["preset"].size,))
        # only using up to 10th order to avoid poor conditioning
        am[:11] = PowerSeriesProfile.from_values(
            s_full, grid_full.compress(data_full["p"]), order=10, sym=False
        ).params

        ai = file.createVariable("ai", np.float64, ("preset",))
        ai.long_name = "rotational transform coefficients"
        ai[:] = np.zeros((file.dimensions["preset"].size,))
        if eq.iota is not None:
            # only using up to 10th order to avoid poor conditioning
            ai[:11] = -PowerSeriesProfile.from_values(  # - for negative Jacobian
                s_full, grid_full.compress(data_full["iota"]), order=10, sym=False
            ).params

        ac = file.createVariable("ac", np.float64, ("preset",))
        ac.long_name = "normalized toroidal current density coefficients"
        ac[:] = np.zeros((file.dimensions["preset"].size,))
        if eq.current is not None:
            # only using up to 10th order to avoid poor conditioning
            ac[:11] = PowerSeriesProfile.from_values(
                s_full, grid_full.compress(data_full["current"]), order=10, sym=False
            ).params

        bdotb = file.createVariable("bdotb", np.float64, ("radius",))
        bdotb.long_name = "flux surface average of magnetic field squared, on full mesh"
        bdotb.units = "T^2"
        bdotb[:] = grid_full.compress(data_full["<|B|^2>"])
        bdotb[0] = 0

        jdotb = file.createVariable("jdotb", np.float64, ("radius",))
        jdotb.long_name = "flux surface average of J*B, on full mesh"
        jdotb.units = "N/m^3"
        jdotb[:] = grid_full.compress(data_full["<J*B>"])
        jdotb[0] = 0

        jcuru = file.createVariable("jcuru", np.float64, ("radius",))
        jcuru.long_name = "flux surface average of sqrt(g)*J^theta, on full mesh"
        jcuru.units = "A/m^3"
        jcuru[:] = -surface_averages(  # - for negative Jacobian
            grid_full,
            data_full["J^theta*sqrt(g)"] / (2 * data_full["rho"]),
            sqrt_g=data_full["sqrt(g)"],
            expand_out=False,
        )
        jcuru[0] = 0

        jcurv = file.createVariable("jcurv", np.float64, ("radius",))
        jcuru.long_name = "flux surface average of sqrt(g)*J^zeta, on full mesh"
        jcurv.units = "A/m^3"
        jcurv[:] = surface_averages(
            grid_full,
            data_full["sqrt(g)"] * data_full["J^zeta"] / (2 * data_full["rho"]),
            sqrt_g=data_full["sqrt(g)"],
            expand_out=False,
        )
        jcurv[0] = 0

        DShear = file.createVariable("DShear", np.float64, ("radius",))
        DShear.long_name = (
            "Mercier stability criterion magnetic shear term, on full mesh"
        )
        DShear.units = "1/Wb^2"
        DShear[:] = grid_full.compress(data_full["D_shear"])
        DShear[0] = 0

        DCurr = file.createVariable("DCurr", np.float64, ("radius",))
        DCurr.long_name = (
            "Mercier stability criterion toroidal current term, on full mesh"
        )
        DCurr.units = "1/Wb^2"
        DCurr[:] = grid_full.compress(data_full["D_current"])
        DCurr[0] = 0

        DWell = file.createVariable("DWell", np.float64, ("radius",))
        DWell.long_name = "Mercier stability criterion magnetic well term, on full mesh"
        DWell.units = "1/Wb^2"
        DWell[:] = grid_full.compress(data_full["D_well"])
        DWell[0] = 0

        DGeod = file.createVariable("DGeod", np.float64, ("radius",))
        DGeod.long_name = (
            "Mercier stability criterion geodesic curvature term, on full mesh"
        )
        DGeod.units = "1/Wb^2"
        DGeod[:] = grid_full.compress(data_full["D_geodesic"])
        DGeod[0] = 0

        DMerc = file.createVariable("DMerc", np.float64, ("radius",))
        DMerc.long_name = "Mercier stability criterion, on full mesh"
        DMerc.units = "1/Wb^2"
        DMerc[:] = grid_full.compress(data_full["D_Mercier"])
        DMerc[0] = 0

        timer.stop("parameters")
        if verbose > 1:
            timer.disp("parameters")

        # independent variables (exact conversion)

        # R axis
        idx = np.where(eq.R_basis.modes[:, 1] == 0)[0]
        R0_n = np.zeros((2 * N + 1,))
        for k in idx:
            (l, m, n) = eq.R_basis.modes[k, :]
            R0_n[n + N] += (-2 * (l // 2 % 2) + 1) * eq.R_lmn[k]
        raxis_cc = file.createVariable("raxis_cc", np.float64, ("n_tor",))
        raxis_cc.long_name = "cos(n*p) component of magnetic axis R coordinate"
        raxis_cc.units = "m"
        raxis_cc[:] = R0_n[N:]
        if not eq.sym:
            raxis_cs = file.createVariable("raxis_cs", np.float64, ("n_tor",))
            raxis_cs.long_name = "sin(n*p) component of magnetic axis R coordinate"
            raxis_cs.units = "m"
            raxis_cs[0] = 0.0
            raxis_cs[1:] = -R0_n[0:N][::-1]

        # Z axis
        idx = np.where(eq.Z_basis.modes[:, 1] == 0)[0]
        Z0_n = np.zeros((2 * N + 1,))
        for k in idx:
            (l, m, n) = eq.Z_basis.modes[k, :]
            Z0_n[n + N] += (-2 * (l // 2 % 2) + 1) * eq.Z_lmn[k]
        zaxis_cs = file.createVariable("zaxis_cs", np.float64, ("n_tor",))
        zaxis_cs.long_name = "sin(n*p) component of magnetic axis Z coordinate"
        zaxis_cs.units = "m"
        zaxis_cs[0] = 0.0
        zaxis_cs[1:] = -Z0_n[0:N][::-1]
        if not eq.sym:
            zaxis_cc = file.createVariable("zaxis_cc", np.float64, ("n_tor",))
            zaxis_cc.long_name = "cos(n*p) component of magnetic axis Z coordinate"
            zaxis_cc.units = "m"
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
        # should negate lambda coefs bc theta_DESC + lambda = theta_PEST,
        # since we are reversing the theta direction (and the theta_PEST direction),
        # so -theta_PEST = -theta_DESC - lambda, so the negative of lambda is what
        # should be saved, so that would be negating all of eq.L_lmn
        # BUT since we are also reversing the poloidal angle direction, which
        # would negate only the coeffs of L_lmn corresponding to m<0
        # (sin theta modes in DESC), the effective result is to only
        # negate the cos(theta) (m>0) lambda modes
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

        sin_basis = DoubleFourierSeries(M=M_nyq, N=N_nyq, NFP=NFP, sym="sin")
        cos_basis = DoubleFourierSeries(M=M_nyq, N=N_nyq, NFP=NFP, sym="cos")
        full_basis = DoubleFourierSeries(M=M_nyq, N=N_nyq, NFP=NFP, sym=False)
        if eq.sym:
            sin_transform = Transform(
                grid=grid_lcfs, basis=sin_basis, build=False, build_pinv=True
            )
            cos_transform = Transform(
                grid=grid_lcfs, basis=cos_basis, build=False, build_pinv=True
            )

            def cosfit(x):
                y = cos_transform.fit(x)
                return np.where(cos_transform.basis.modes[:, 1] < 0, -y, y)

            def sinfit(x):
                y = sin_transform.fit(x)
                return np.where(sin_transform.basis.modes[:, 1] < 0, -y, y)

        else:
            full_transform = Transform(
                grid=grid_lcfs, basis=full_basis, build=False, build_pinv=True
            )

            def fullfit(x):
                y = full_transform.fit(x)
                return np.where(full_transform.basis.modes[:, 1] < 0, -y, y)

        rmin_surf = file.createVariable("rmin_surf", np.float64)
        rmin_surf.long_name = "minimum R coordinate range"
        rmin_surf.units = "m"
        rmin_surf[:] = np.amin(data_lcfs["R"])

        rmax_surf = file.createVariable("rmax_surf", np.float64)
        rmax_surf.long_name = "maximum R coordinate range"
        rmax_surf.units = "m"
        rmax_surf[:] = np.amax(data_lcfs["R"])

        zmax_surf = file.createVariable("zmax_surf", np.float64)
        zmax_surf.long_name = "maximum Z coordinate range"
        zmax_surf.units = "m"
        zmax_surf[:] = np.amax(np.abs(data_lcfs["Z"]))

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
            (data_half["sqrt(g)"] / (2 * data_half["rho"]))
            .reshape(
                (grid_half.num_theta, grid_half.num_rho, grid_half.num_zeta), order="F"
            )
            .transpose((1, 0, 2))
            .reshape((grid_half.num_rho, -1), order="F")
        )
        x_mn = np.zeros((surfs - 1, m.size))
        for i in range(surfs - 1):
            if eq.sym:
                x_mn[i, :] = cosfit(data[i, :])
            else:
                x_mn[i, :] = fullfit(data[i, :])
        xm, xn, s, c = ptolemy_identity_rev(m, n, x_mn)
        gmnc[0, :] = 0
        gmnc[1:, :] = -c  # negative sign for negative Jacobian
        if not eq.sym:
            gmns[0, :] = 0
            gmns[1:, :] = -s
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
            data_half["|B|"]
            .reshape(
                (grid_half.num_theta, grid_half.num_rho, grid_half.num_zeta), order="F"
            )
            .transpose((1, 0, 2))
            .reshape((grid_half.num_rho, -1), order="F")
        )
        x_mn = np.zeros((surfs - 1, m.size))
        for i in range(surfs - 1):
            if eq.sym:
                x_mn[i, :] = cosfit(data[i, :])
            else:
                x_mn[i, :] = fullfit(data[i, :])
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
            data_half["B^theta"]
            .reshape(
                (grid_half.num_theta, grid_half.num_rho, grid_half.num_zeta), order="F"
            )
            .transpose((1, 0, 2))
            .reshape((grid_half.num_rho, -1), order="F")
        )
        x_mn = np.zeros((surfs - 1, m.size))
        for i in range(surfs - 1):
            if eq.sym:
                x_mn[i, :] = cosfit(data[i, :])
            else:
                x_mn[i, :] = fullfit(data[i, :])
        xm, xn, s, c = ptolemy_identity_rev(m, n, x_mn)
        bsupumnc[0, :] = 0
        bsupumnc[1:, :] = -c  # negative sign for negative Jacobian
        if not eq.sym:
            bsupumns[0, :] = 0
            bsupumns[1:, :] = -s
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
            data_half["B^zeta"]
            .reshape(
                (grid_half.num_theta, grid_half.num_rho, grid_half.num_zeta), order="F"
            )
            .transpose((1, 0, 2))
            .reshape((grid_half.num_rho, -1), order="F")
        )
        x_mn = np.zeros((surfs - 1, m.size))
        for i in range(surfs - 1):
            if eq.sym:
                x_mn[i, :] = cosfit(data[i, :])
            else:
                x_mn[i, :] = fullfit(data[i, :])
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
        data = data_full["B_rho"].reshape(
            (grid_full.num_theta, grid_full.num_rho, grid_full.num_zeta), order="F"
        ).transpose((1, 0, 2)).reshape((grid_full.num_rho, -1), order="F") / (
            2 * r_full[:, np.newaxis]
        )
        # B_rho -> B_psi conversion: d(rho)/d(s) = 1/(2*rho)
        x_mn = np.zeros((surfs, m.size))
        for i in range(surfs):
            if eq.sym:
                x_mn[i, :] = sinfit(data[i, :])
            else:
                x_mn[i, :] = fullfit(data[i, :])
        xm, xn, s, c = ptolemy_identity_rev(m, n, x_mn)
        bsubsmns[:, :] = s
        bsubsmns[0, :] = (  # linear extrapolation for coefficient at the magnetic axis
            s[1, :] - (s[2, :] - s[1, :]) / (s_full[2] - s_full[1]) * s_full[1]
        )
        # TODO (#1379): evaluate current at rho=0 nodes instead of extrapolation
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
        data = (
            data_half["B_theta"]
            .reshape(
                (grid_half.num_theta, grid_half.num_rho, grid_half.num_zeta), order="F"
            )
            .transpose((1, 0, 2))
            .reshape((grid_half.num_rho, -1), order="F")
        )
        x_mn = np.zeros((surfs - 1, m.size))
        for i in range(surfs - 1):
            if eq.sym:
                x_mn[i, :] = cosfit(data[i, :])
            else:
                x_mn[i, :] = fullfit(data[i, :])
        xm, xn, s, c = ptolemy_identity_rev(m, n, x_mn)
        bsubumnc[0, :] = 0
        bsubumnc[1:, :] = -c  # negative sign for negative Jacobian
        if not eq.sym:
            bsubumns[0, :] = 0
            bsubumns[1:, :] = -s
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
            data_half["B_zeta"]
            .reshape(
                (grid_half.num_theta, grid_half.num_rho, grid_half.num_zeta), order="F"
            )
            .transpose((1, 0, 2))
            .reshape((grid_half.num_rho, -1), order="F")
        )
        x_mn = np.zeros((surfs - 1, m.size))
        for i in range(surfs - 1):
            if eq.sym:
                x_mn[i, :] = cosfit(data[i, :])
            else:
                x_mn[i, :] = fullfit(data[i, :])
        xm, xn, s, c = ptolemy_identity_rev(m, n, x_mn)
        bsubvmnc[0, :] = 0
        bsubvmnc[1:, :] = c
        if not eq.sym:
            bsubvmns[0, :] = 0
            bsubvmns[1:, :] = s
        timer.stop("B_zeta")
        if verbose > 1:
            timer.disp("B_zeta")

        # J^theta
        timer.start("J^theta*sqrt(g)")
        if verbose > 0:
            print("Saving J^theta*sqrt(g)")
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
            (data_full["J^theta*sqrt(g)"] / (2 * data_full["rho"]))
            .reshape(
                (grid_full.num_theta, grid_full.num_rho, grid_full.num_zeta), order="F"
            )
            .transpose((1, 0, 2))
            .reshape((grid_full.num_rho, -1), order="F")
        )
        x_mn = np.zeros((surfs, m.size))
        for i in range(1, surfs):  # skip NaN values at magnetic axis
            if eq.sym:
                x_mn[i, :] = cosfit(data[i, :])
            else:
                x_mn[i, :] = fullfit(data[i, :])
        xm, xn, s, c = ptolemy_identity_rev(m, n, x_mn)
        currumnc[:, :] = c
        currumnc[0, :] = (  # linear extrapolation for coefficient at the magnetic axis
            s[1, :] - (c[2, :] - c[1, :]) / (s_full[2] - s_full[1]) * s_full[1]
        )
        # TODO (#1379): evaluate current at rho=0 nodes instead of extrapolation
        if not eq.sym:
            currumns[:, :] = s
            currumns[0, :] = (
                s[1, :] - (s[2, :] - s[1, :]) / (s_full[2] - s_full[1]) * s_full[1]
            )
        timer.stop("J^theta*sqrt(g)")
        if verbose > 1:
            timer.disp("J^theta*sqrt(g)")

        # J^zeta
        timer.start("J^zeta*sqrt(g)")
        if verbose > 0:
            print("Saving J^zeta*sqrt(g)")
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
            (data_full["J^zeta"] * data_full["sqrt(g)"] / (2 * data_full["rho"]))
            .reshape(
                (grid_full.num_theta, grid_full.num_rho, grid_full.num_zeta), order="F"
            )
            .transpose((1, 0, 2))
            .reshape((grid_full.num_rho, -1), order="F")
        )
        x_mn = np.zeros((surfs, m.size))
        for i in range(1, surfs):  # skip NaN values at magnetic axis
            if eq.sym:
                x_mn[i, :] = cosfit(data[i, :])
            else:
                x_mn[i, :] = fullfit(data[i, :])
        xm, xn, s, c = ptolemy_identity_rev(m, n, x_mn)
        currvmnc[:, :] = -c  # negative sign for negative Jacobian
        currvmnc[0, :] = -(  # linear extrapolation for coefficient at the magnetic axis
            s[1, :] - (c[2, :] - c[1, :]) / (s_full[2] - s_full[1]) * s_full[1]
        )
        # TODO (#1379): evaluate current at rho=0 nodes instead of extrapolation
        if not eq.sym:
            currvmns[:, :] = -s
            currumns[0, :] = -(
                s[1, :] - (s[2, :] - s[1, :]) / (s_full[2] - s_full[1]) * s_full[1]
            )
        timer.stop("J^zeta*sqrt(g)")
        if verbose > 1:
            timer.disp("J^zeta*sqrt(g)")

        # TODO (#1380): these output quantities need to be added
        bdotgradv = file.createVariable("bdotgradv", np.float64, ("radius",))
        bdotgradv[:] = np.zeros((file.dimensions["radius"].size,))
        bdotgradv.long_name = "Not Implemented: This output is hard-coded to 0!"
        bdotgradv.units = "None"
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

        extcur = file.createVariable("extcur", np.float64)
        extcur.long_name = "external current?"
        extcur[:] = np.nan  # VMEC gives nothing for fixed-boundary solutions

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

        niter = file.createVariable("niter", np.int32)
        niter[:] = 1

        over_r = file.createVariable("over_r", np.float64, ("radius",))
        over_r[:] = np.zeros((file.dimensions["radius"].size,))

        specw = file.createVariable("specw", np.float64, ("radius",))
        specw[:] = np.zeros((file.dimensions["radius"].size,))

        wdot = file.createVariable("wdot", np.float64, ("time",))
        wdot[:] = np.zeros((file.dimensions["time"].size,))
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
            Filename of VMEC output file.

        Returns
        -------
        vmec_data : dict
            The VMEC data fields.

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

    @classmethod
    def write_vmec_input(cls, eq, fname, header="", **kwargs):  # noqa: C901
        """Write a VMEC input file for an equivalent DESC equilibrium.

        Parameters
        ----------
        eq : Equilibrium
            Equilibrium to write the input file for.
        fname : str or path-like
            Filename of VMEC input file.
        header : str
            Text to print at the top of the file.

        Returns
        -------
        None

        """
        # open the file, unless its already open
        if not isinstance(fname, io.IOBase):
            f = open(fname, "w+")
        else:
            f = fname
        f.seek(0)

        f.write("! " + header + "\n")
        f.write("&INDATA\n")

        # free-boundary parameters (values are hard-coded for fixed-boundary solve)
        f.write("!---- Free-Boundary Parameters ----\n")
        f.write("  LFREEB = F\n")  # free-boundary?
        f.write("  MGRID_FILE = 'none'\n")  # MGRID file name
        f.write("  NVACSKIP = 5\n")  # how often to update vaccum solution

        f.write("!---- Runtime Parameters ----\n")
        # how often (in iterations) to print diagnostics
        f.write("  NSTEP = {:3.0f}\n".format(kwargs.get("NITER", 250)))
        # blending parameter from previous iteration [0, 1]
        f.write("  DELT = {}\n".format(kwargs.get("DELT", 0.9)))
        f.write("  NS_ARRAY =   ")  # number of flux surfaces
        for ns in kwargs.get("NS_ARRAY", [17, 33, 65, 129, 257]):
            f.write(" {:5.0f}".format(ns))
        f.write("\n  NITER_ARRAY =")  # maximum number of iterations
        for niter in kwargs.get("NITER_ARRAY", [1e4, 2e4, 3e4, 4e4, 5e4]):
            f.write(" {:5.0f}".format(niter))
        f.write("\n  FTOL_ARRAY = ")  # stopping tolerance
        for ftol in kwargs.get("FTOL_ARRAY", [1e-12, 1e-12, 1e-12, 1e-12, 1e-12]):
            f.write(" {:5.0E}".format(ftol))
        f.write("\n")

        f.write("!---- Grid Parameters ----\n")
        f.write("  LASYM = {}\n".format("F" if eq.sym else "T"))  # stellarator symmetry
        f.write("  NFP = {:2.0f}\n".format(eq.NFP))  # number of field periods
        # poloidal resolution
        f.write("  MPOL = {:2.0f}\n".format(kwargs.get("MPOL", eq.M + 1)))
        # toroidal resolution
        f.write("  NTOR = {:2.0f}\n".format(kwargs.get("NTOR", eq.N)))
        # total toroidal magnetic flux (Wb)
        f.write("  PHIEDGE = {:+14.8E}\n".format(eq.Psi))

        f.write("!---- Pressure Parameters ----\n")
        f.write("  GAMMA = 0\n")  # pressure profile specified
        f.write("  PRES_SCALE = {}\n".format(kwargs.get("PRES_SCALE", 1)))  # AM scale
        if eq.pressure is not None:
            pressure = eq.pressure
        else:
            # if kinetic profiles, fit pressure to power series
            grid = LinearGrid(L=eq.L_grid, axis=True)
            data = eq.compute(["rho", "p"], grid=grid)
            rho = grid.compress(data["rho"])
            p = grid.compress(data["p"])
            pressure = PowerSeriesProfile.from_values(rho, p, order=eq.L, sym=True)
        if isinstance(pressure, PowerSeriesProfile) and pressure.sym:
            f.write("  AM =")  # pressure power series coefficients
            for am in pressure.params:
                f.write(" {:+14.8E}".format(am))
            f.write("\n  PMASS_TYPE = 'power_series'\n")
        elif isinstance(pressure, PowerSeriesProfile) and not pressure.sym:
            rho = np.linspace(0, 1, pressure.basis.L + 1)
            pressure = SplineProfile(values=pressure(rho), knots=rho)
        if isinstance(pressure, SplineProfile):
            f.write("  AM_AUX_S =")  # spline knot locations
            for r in pressure.knots:
                f.write(" {:+14.8E}".format(r**2))  # s = rho^2
            f.write("\n  AM_AUX_F =")  # pressure cubic spline values
            for am in pressure.params:
                f.write(" {:+14.8E}".format(am))
            f.write("\n  PMASS_TYPE = 'cubic_spline'\n")

        f.write("!---- Current/Iota Parameters ----\n")
        if eq.current is None:
            iota = eq.iota
            f.write("  NCURR = 0\n")  # rotational transform profile specified
            if isinstance(iota, PowerSeriesProfile) and iota.sym:
                f.write("  AI =")  # iota power series coefficients
                for ai in iota.params:
                    f.write(" {:+14.8E}".format(ai))
                f.write("\n  PIOTA_TYPE = 'power_series'\n")
            elif isinstance(iota, PowerSeriesProfile) and not iota.sym:
                rho = np.linspace(0, 1, iota.basis.L + 1)
                iota = SplineProfile(values=iota(rho), knots=rho)
            if isinstance(iota, SplineProfile):
                f.write("  AI_AUX_S =")  # spline knot locations
                for r in iota.knots:
                    f.write(" {:+14.8E}".format(r**2))  # s = rho^2
                f.write("\n  AI_AUX_F =")  # iota cubic spline values
                for ai in iota.params:
                    f.write(" {:+14.8E}".format(ai))
                f.write("\n  PIOTA_TYPE = 'cubic_spline'\n")
        else:
            current = eq.current
            f.write("  NCURR = 1\n")  # current profile specified
            f.write("  CURTOR = {:+14.8E}\n".format(float(current(1)[0])))  # AC scale
            if isinstance(current, PowerSeriesProfile) and current.sym:
                f.write("  AC =")  # current power series coefficients
                for ac in current.params:
                    f.write(" {:+14.8E}".format(ac))
                f.write("\n  PCURR_TYPE = 'power_series_I'\n")
            else:
                rho = np.linspace(0, 1, eq.L_grid + 1)
                f.write("  AC_AUX_S =")  # spline knot locations
                for r in rho:
                    f.write(" {:+14.8E}".format(r**2))  # s = rho^2
                f.write("\n  AC_AUX_F =")  # current cubic spline values
                for r in rho:
                    # dI/ds = dI/drho / (2*rho)
                    f.write(
                        " {:+14.8E}".format(
                            0
                            if np.abs(r) < np.finfo(r.dtype).eps
                            else float(current(r, dr=1) / (2 * r))
                        )
                    )
                f.write("\n  PCURR_TYPE = 'cubic_spline_Ip'\n")

        f.write("!---- Axis Parameters ----\n")
        # R axis
        idx = np.where(eq.R_basis.modes[:, 1] == 0)[0]
        R0_n = np.zeros((2 * eq.N + 1,))
        for k in idx:
            (l, m, n) = eq.R_basis.modes[k, :]
            R0_n[n + eq.N] += (-2 * (l // 2 % 2) + 1) * eq.R_lmn[k]
        # Z axis
        idx = np.where(eq.Z_basis.modes[:, 1] == 0)[0]
        Z0_n = np.zeros((2 * eq.N + 1,))
        for k in idx:
            (l, m, n) = eq.Z_basis.modes[k, :]
            Z0_n[n + eq.N] += (-2 * (l // 2 % 2) + 1) * eq.Z_lmn[k]
        # R axis cosine coefficients
        f.write("  RAXIS_CC = ")
        for rac in R0_n[eq.N :]:
            f.write("{:+14.8E} ".format(rac))
        if not eq.sym:
            # R axis sine coefficients
            f.write("\n  RAXIS_CS = {:+14.8E}".format(0))
            for ras in -R0_n[0 : eq.N][::-1]:
                f.write(" {:+14.8E}".format(ras))
            # Z axis cosine coefficients
            f.write("\n  ZAXIS_CC =")
            for zac in Z0_n[eq.N :]:
                f.write(" {:+14.8E}".format(zac))
        # Z axis sine coefficients
        f.write("\n  ZAXIS_CS = {:+14.8E}".format(0))
        for zas in -Z0_n[0 : eq.N][::-1]:
            f.write(" {:+14.8E}".format(zas))
        f.write("\n")

        f.write("!---- Boundary Parameters ----\n")
        # R boundary coefficients
        M, N, RBS, RBC = ptolemy_identity_rev(
            eq.surface.R_basis.modes[:, 1],
            eq.surface.R_basis.modes[:, 2],
            eq.surface.R_lmn,
        )
        # Z boundary coefficients
        _, _, ZBS, ZBC = ptolemy_identity_rev(
            eq.surface.Z_basis.modes[:, 1],
            eq.surface.Z_basis.modes[:, 2],
            eq.surface.Z_lmn,
        )
        if eq.sym:
            for m, n, rbc, zbs in np.vstack(
                (np.atleast_2d(M), np.atleast_2d(N), RBC, ZBS)
            ).T:
                f.write(
                    f"  RBC({n:3.0f},{m:3.0f}) = {rbc:+14.8E}"
                    + f"  ZBS({n:3.0f},{m:3.0f}) = {zbs:+14.8E}\n"
                )
        else:
            for m, n, rbc, rbs, zbc, zbs in np.vstack(
                (np.atleast_2d(M), np.atleast_2d(N), RBC, RBS, ZBC, ZBS)
            ).T:
                f.write(
                    f"  RBC({n:3.0f},{m:3.0f}) = {rbc:+14.8E}"
                    + f"  RBS({n:3.0f},{m:3.0f}) = {rbs:+14.8E}"
                    + f"  ZBC({n:3.0f},{m:3.0f}) = {zbc:+14.8E}"
                    + f"  ZBS({n:3.0f},{m:3.0f}) = {zbs:+14.8E}\n"
                )

        f.write("/")
        f.close()
        return None

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
    def compute_theta_coords(
        cls, lmns, xm, xn, s, theta_star, zeta, si=None, lmnc=None
    ):
        """Find θ (theta_DESC) for given PEST straight field line ϑ (theta_star).

        Parameters
        ----------
        lmns : array-like
            sin(mt-nz) Fourier coefficients for lambda
        xm : array-like
            poloidal mode numbers
        xn : array-like
            toroidal mode numbers
        s : array-like
            desired radial coordinates (normalized toroidal magnetic flux)
        theta_star : array-like
            desired straight field-line poloidal angles (PEST/VMEC-like flux
            coordinates)
        zeta : array-like
            desired toroidal angles (toroidal coordinate phi)
        si : ndarray
            values of radial coordinates where lmns are defined. Defaults to linearly
            spaced on half grid between (0,1)
        lmnc : array-like, optional
            cos(mt-nz) Fourier coefficients for lambda

        Returns
        -------
        theta : ndarray
            theta such that theta + lambda(theta) == theta_star

        """
        theta_PEST = theta_star
        if si is None:
            si = np.linspace(0, 1, lmns.shape[0])
            si[1:] = si[0:-1] + 0.5 / (lmns.shape[0] - 1)
        lmbda_mns = interpolate.CubicSpline(si, lmns)
        if lmnc is None:
            lmbda_mnc = lambda s: 0
        else:
            lmbda_mnc = interpolate.CubicSpline(si, lmnc)

        # Root finding for θₖ such that r(θₖ) = ϑₖ(ρ, θₖ, ζ) − ϑ = 0.
        def root_fun(theta):
            lmbda = np.sum(
                lmbda_mns(s)
                * np.sin(
                    xm[np.newaxis] * theta[:, np.newaxis]
                    - xn[np.newaxis] * zeta[:, np.newaxis]
                ),
                axis=-1,
            ) + np.sum(
                lmbda_mnc(s)
                * np.cos(
                    xm[np.newaxis] * theta[:, np.newaxis]
                    - xn[np.newaxis] * zeta[:, np.newaxis]
                ),
                axis=-1,
            )
            theta_PEST_k = theta + lmbda
            r = theta_PEST_k - theta_PEST
            return -r  # the negative sign is necessary

        out = optimize.root(
            root_fun, x0=theta_PEST, method="diagbroyden", options={"ftol": 1e-6}
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

        # straight field line angles to plot
        tr = np.linspace(0, 1, 50)
        tt = np.linspace(0, 2 * np.pi, Nt, endpoint=False)
        tz = np.linspace(0, 2 * np.pi / equil.NFP, Nz, endpoint=False)
        t_grid = LinearGrid(rho=tr, theta=tt, zeta=tz, NFP=equil.NFP)

        # Note: theta* (also known as vartheta) is the poloidal straight field line
        # angle in PEST-like flux coordinates

        # find theta angles corresponding to desired theta* angles
        v_grid = Grid(
            equil.map_coordinates(t_grid.nodes, inbasis=("rho", "theta_PEST", "zeta"))
        )
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

        sym = "lmnc" not in vmec_data.keys()

        v_nodes = cls.compute_theta_coords(
            vmec_data["lmns"],
            vmec_data["xm"],
            vmec_data["xn"],
            t_nodes[:, 0],
            t_nodes[:, 1],
            t_nodes[:, 2],
            lmnc=vmec_data["lmnc"] if not sym else None,
        )

        t_nodes[:, 1] = v_nodes
        if sym:
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
        else:
            Rr_vmec = cls.vmec_interpolate(
                vmec_data["rmnc"],
                vmec_data["rmns"],
                vmec_data["xm"],
                vmec_data["xn"],
                theta=r_nodes[:, 1],
                phi=r_nodes[:, 2],
                s=r_nodes[:, 0],
                sym=False,
            )
            Zr_vmec = cls.vmec_interpolate(
                vmec_data["zmnc"],
                vmec_data["zmns"],
                vmec_data["xm"],
                vmec_data["xn"],
                theta=r_nodes[:, 1],
                phi=r_nodes[:, 2],
                s=r_nodes[:, 0],
                sym=False,
            )
            Rv_vmec = cls.vmec_interpolate(
                vmec_data["rmnc"],
                vmec_data["rmns"],
                vmec_data["xm"],
                vmec_data["xn"],
                theta=t_nodes[:, 1],
                phi=t_nodes[:, 2],
                s=t_nodes[:, 0],
                sym=False,
            )
            Zv_vmec = cls.vmec_interpolate(
                vmec_data["zmnc"],
                vmec_data["zmns"],
                vmec_data["xm"],
                vmec_data["xn"],
                theta=t_nodes[:, 1],
                phi=t_nodes[:, 2],
                s=t_nodes[:, 0],
                sym=False,
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
