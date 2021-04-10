import os
import math
import numpy as np

from netCDF4 import Dataset, stringtochar
from shapely.geometry import Polygon
import matplotlib.pyplot as plt

from desc.grid import Grid, LinearGrid
from desc.basis import DoubleFourierSeries
from desc.transform import Transform
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
        inputs["Psi"] = file.variables["phi"][-1]
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
        BC = LCFSConstraint(
            eq.R_basis,
            eq.Z_basis,
            eq.L_basis,
            eq.Rb_basis,
            eq.Zb_basis,
            eq.Rb_lmn,
            eq.Zb_lmn,
        )
        eq.x = BC.make_feasible(eq.x)

        return eq

    @classmethod
    def save(cls, eq, path, surfs=128):
        """Save an Equilibrium as a netCDF file in the VMEC format.

        Parameters
        ----------
        eq : Equilibrium
            Equilibrium to save.
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
        m, n, x_mn = zernike_to_fourier(eq.R_lmn, basis=eq.R_basis, rho=r_full)
        xm, xn, s, c = ptolemy_identity_rev(m, n, x_mn)
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
        m, n, x_mn = zernike_to_fourier(eq.Z_lmn, basis=eq.Z_basis, rho=r_full)
        xm, xn, s, c = ptolemy_identity_rev(m, n, x_mn)
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
        m, n, x_mn = zernike_to_fourier(eq.L_lmn, basis=eq.L_basis, rho=r_half)
        xm, xn, s, c = ptolemy_identity_rev(m, n, x_mn)
        lmns[1:, :] = s
        if not eq.sym:
            lmnc[1:, :] = c

        # derived quantities (approximate conversion)

        MM = 2 * math.ceil(1.5 * M) + 1
        NN = 2 * math.ceil(1.5 * N) + 1
        grid = LinearGrid(M=MM, N=NN, NFP=NFP)

        if eq.sym:
            sin_basis = DoubleFourierSeries(M=M, N=N, NFP=NFP, sym="sin")
            cos_basis = DoubleFourierSeries(M=M, N=N, NFP=NFP, sym="cos")
            sin_transform = Transform(grid=grid, basis=sin_basis, build_pinv=True)
            cos_transform = Transform(grid=grid, basis=cos_basis, build_pinv=True)
        else:
            full_basis = DoubleFourierSeries(M=M, N=N, NFP=NFP, sym=None)
            full_transform = Transform(grid=grid, basis=full_basis, build_pinv=True)

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
        xm, xn, s, c = ptolemy_identity_rev(m, n, x_mn)
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
        xm, xn, s, c = ptolemy_identity_rev(m, n, x_mn)
        bmnc[1:, :] = c
        if not eq.sym:
            bmns[1:, :] = s

        file.close

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
    def vmec_interpolate(Cmn, Smn, xm, xn, theta, phi, lam=None, sym=True):
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
        theta : ndarray, shape(Nt,Nz)
            poloidal angles
        phi : ndarray, shape(Nt,Nz)
            toroidal angles
        sym : bool
            stellarator symmetry (Default value = True)

        Returns
        -------
        if sym = True
            C, S (tuple of ndarray): VMEC data interpolated at the angles (theta,phi)
            where C has cosine symmetry and S has sine symmetry
        if sym = False
            X (ndarray): non-symmetric VMEC data interpolated at the angles (theta,phi)

        """
        C_arr = []
        S_arr = []
        dim = Cmn.shape
        if lam is None:
            lam = np.zeros((dim[0], *theta.shape))

        for j in range(dim[1]):

            m = xm[j]
            n = xn[j]

            C = [
                Cmn[s, j] * np.cos(m * (theta - lam[s]) - n * phi)
                for s in range(dim[0])
            ]
            S = [
                Smn[s, j] * np.sin(m * (theta - lam[s]) - n * phi)
                for s in range(dim[0])
            ]
            C_arr.append(C)
            S_arr.append(S)

        C = np.sum(C_arr, axis=0)
        S = np.sum(S_arr, axis=0)
        if sym:
            return C, S
        else:
            return C + S

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
                Polygon(np.array([R[:, i], Z[:, i]]).T)
                for R, Z in zip(coords["Rr_vmec"], coords["Zr_vmec"])
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
        """Compute ???.

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
        rr = np.sqrt(idxes / Nr_vmec)
        rt = np.linspace(0, 2 * np.pi, num_theta)
        rz = np.linspace(0, 2 * np.pi / equil.NFP, Nz)
        r_grid = LinearGrid(rho=rr, theta=rt, zeta=rz)
        tr = np.linspace(0, 1, 50)
        tt = np.linspace(0, 2 * np.pi, Nt, endpoint=False)
        tz = np.linspace(0, 2 * np.pi / equil.NFP, Nz)
        t_grid = LinearGrid(rho=tr, theta=tt, zeta=tz)

        r_coords_desc = equil.compute_toroidal_coords(r_grid)
        t_coords_desc = equil.compute_toroidal_coords(t_grid)

        # theta coordinates cooresponding to linearly spaced vartheta angles
        v_nodes = t_grid.nodes
        v_nodes[:, 1] = t_grid.nodes[:, 1] - t_coords_desc["lambda"]
        v_grid = Grid(v_nodes)
        v_coords_desc = equil.compute_toroidal_coords(v_grid)

        # r contours
        Rr_desc = r_coords_desc["R"].reshape((r_grid.M, r_grid.L, r_grid.N), order="F")
        Zr_desc = r_coords_desc["Z"].reshape((r_grid.M, r_grid.L, r_grid.N), order="F")

        # theta contours
        Rv_desc = v_coords_desc["R"].reshape((t_grid.M, t_grid.L, t_grid.N), order="F")
        Zv_desc = v_coords_desc["Z"].reshape((t_grid.M, t_grid.L, t_grid.N), order="F")

        rtt, rzz = np.meshgrid(rt, rz, indexing="ij")
        ttt, tzz = np.meshgrid(tt, tz, indexing="ij")

        _, L_vmec = cls.vmec_interpolate(
            np.zeros_like(vmec_data["lmns"]),
            vmec_data["lmns"],
            vmec_data["xm"],
            vmec_data["xn"],
            ttt,
            tzz,
        )

        Rr_vmec, Zr_vmec = cls.vmec_interpolate(
            vmec_data["rmnc"][idxes],
            vmec_data["zmns"][idxes],
            vmec_data["xm"],
            vmec_data["xn"],
            rtt,
            rzz,
        )

        Rv_vmec, Zv_vmec = cls.vmec_interpolate(
            vmec_data["rmnc"],
            vmec_data["zmns"],
            vmec_data["xm"],
            vmec_data["xn"],
            ttt,
            tzz,
            lam=L_vmec,
        )

        coords = {
            "Rr_desc": Rr_desc,
            "Zr_desc": Zr_desc,
            "Rv_desc": Rv_desc,
            "Zv_desc": Zv_desc,
            "Rr_vmec": Rr_vmec,
            "Zr_vmec": Zr_vmec,
            "Rv_vmec": Rv_vmec,
            "Zv_vmec": Zv_vmec,
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
                coords["Rr_vmec"][:, :, k].T, coords["Zr_vmec"][:, :, k].T, "b-"
            )
            ax[k].plot(coords["Rv_vmec"][:, :, k], coords["Zv_vmec"][:, :, k], "b-")

            ax[k].plot(coords["Rr_desc"][0, 0, k], coords["Zr_desc"][0, 0, k], "ro")
            ax[k].plot(coords["Rv_desc"][:, :, k].T, coords["Zv_desc"][:, :, k].T, "r:")
            s_desc = ax[k].plot(
                coords["Rr_desc"][:, :, k], coords["Zr_desc"][:, :, k], "r:"
            )

            ax[k].axis("equal")
            ax[k].set_xlabel("R")
            ax[k].set_ylabel("Z")
            ax[k].grid(True)
            if k == 0:
                s_vmec[0].set_label("VMEC")
                s_desc[0].set_label("DESC")
                ax[k].legend(fontsize="xx-small")

        return fig, ax
