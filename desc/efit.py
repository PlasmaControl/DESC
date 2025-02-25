"""Class for reading and processing EFIT files."""

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline

from desc.basis import DoubleFourierSeries
from desc.equilibrium import Equilibrium
from desc.geometry import FourierRZToroidalSurface
from desc.grid import LinearGrid
from desc.profiles import PowerSeriesProfile, SplineProfile
from desc.transform import Transform


class EFITIO:
    """Performs input from EFIT files to DESC Equilibrium."""

    @classmethod
    def load(  # noqa: C901 - fxn too complex
        cls,
        g,
        M,
        N=0,
        L=None,
        bdry_dist=1,
    ):
        """Create a DESC equilibrium object from an EFIT geqdsk file.

        The equilibrium will not be solved, this just creates an equilibirum
        with the correct boundary and profiles.

        Parameters
        ----------
        g : path-like or dict
            Path to geqdsk file, or dict of already read parameters.
        M : int
            Poloidal resolution of the equilibrium. Used for fitting the boundary.
        N : int, optional
            Toroidal resolution of the equilibrium.
        L : int, optional
            Radial resolution of the equilibrium. Used for fitting profiles.
            If None, spline profiles will be used. Otherwise they are fit
            to even power series in rho.
        bdry_dist : float
            Deviation from separatrix (boundary distance) in terms of the
            normalized poloidal flux. If bdry_dist == 1, then the boundary
            is the actual separatrix read from the geqdsk file.
            bdry_dist = 0 corresponds to the magnetic axis.
            Typical values should be 0.95 < bdry_dist < 1.0

        Returns
        -------
        eq : Equilibrium
            DESC equilibrium with boundary shape, pressure, iota, and
            total toroidal flux from geqdsk file.
        """
        if not isinstance(g, dict):
            with open(g) as f:
                lines = f.readlines()
            g = {}

            has_spaces = all(
                [elem == 5 for elem in [len(lines[i].split()) for i in range(1, 5)]]
            )
            if has_spaces:

                def splitline(s):
                    return s.split()

                def splitarr(l):
                    return " ".join(l).split()

            else:

                def splitline(s):
                    ss = s.split("\n")[0]
                    idx = np.arange(0, len(ss) + 1, 16)
                    return [ss[i:j] for i, j in zip(idx[:-1], idx[1:])]

                def splitarr(l):
                    return splitline("".join([line.split("\n")[0] for line in l]))

            line0 = lines[0].split()

            # points in the horizontal direction (width) and vertical direction (height)
            g["nw"] = int("".join([c for c in line0[-2] if c.isdigit()]))
            g["nh"] = int("".join([c for c in line0[-1] if c.isdigit()]))
            if len(line0) == 8:  # EFIT file information
                g["efit"] = line0[4]
            else:
                g["efit"] = None

            [g["rdim"], g["zdim"], g["rcentr"], g["rleft"], g["zmid"]] = [
                float(foo) for foo in splitline(lines[1])
            ]
            [
                g["rmaxis"],
                g["zmaxis"],
                g["chi_axis"],
                g["chi_boundary"],
                g["bcentr"],
            ] = [float(foo) for foo in splitline(lines[2])]
            [g["zmaxis"], _, g["chi_boundary"], _, _] = [
                float(foo) for foo in splitline(lines[4])
            ]
            lines_per_profile = int(np.ceil(g["nw"] / 5))
            lines_psirz = int(np.ceil(g["nw"] * g["nh"] / 5))
            lines = lines[5:]

            # All the flux functions are stored in first few rows of the file
            g["fpol"] = np.array(
                [float(foo) for foo in splitarr(lines[:lines_per_profile])]
            )
            lines = lines[
                lines_per_profile:
            ]  # redefine lines to remove the lines we just read

            g["pres"] = np.array(
                [float(foo) for foo in splitarr(lines[:lines_per_profile])]
            )
            lines = lines[lines_per_profile:]

            g["ffprime"] = np.array(
                [float(foo) for foo in splitarr(lines[:lines_per_profile])]
            )
            lines = lines[lines_per_profile:]

            g["pprime"] = np.array(
                [float(foo) for foo in splitarr(lines[:lines_per_profile])]
            )
            lines = lines[lines_per_profile:]

            # Read the poloidal flux stored as a 2D array
            g["chirz"] = np.array(
                [float(foo) for foo in splitarr(lines[:lines_psirz])]
            ).reshape((g["nh"], g["nw"]))
            lines = lines[lines_psirz:]

            # Safety factor
            g["q"] = np.array(
                [float(foo) for foo in splitarr(lines[:lines_per_profile])]
            )
            lines = lines[lines_per_profile:]

            if bdry_dist == 1:  # distance of the boundary from the axis
                if len(lines) > 0:
                    [g["nbbbs"], g["limitr"]] = [int(foo) for foo in lines[0].split()]
                    lines = lines[1:]
                    lines_nbbbs = int(np.ceil(2 * g["nbbbs"] / 5))
                    lines_limitr = int(np.ceil(2 * g["limitr"] / 5))
                    if len(lines) >= lines_nbbbs:
                        rzbbbs = np.array(
                            [float(foo) for foo in splitarr(lines[:lines_nbbbs])]
                        ).reshape((g["nbbbs"], 2))
                        g["rbbbs"], g["zbbbs"] = rzbbbs[:, 0], rzbbbs[:, 1]
                        lines = lines[lines_nbbbs:]
                    if len(lines) >= lines_limitr:
                        rzlimitr = np.array(
                            [float(foo) for foo in splitarr(lines[:lines_limitr])]
                        ).reshape((g["limitr"], 2))
                        g["rlimitr"], g["zlimitr"] = rzlimitr[:, 0], rzlimitr[:, 1]
                    lines = lines[lines_limitr:]

            else:
                Rmin = g["rleft"]
                Rmax = Rmin + g["rdim"]
                Rgrid = np.linspace(Rmin, Rmax, g["nw"])

                # zmid is the same as zmaxis
                Zmin = g["zmid"] - g["zdim"] / 2
                Zmax = g["zmid"] + g["zdim"] / 2
                Zgrid = np.linspace(Zmin, Zmax, g["nh"])

                RR, ZZ = np.meshgrid(Rgrid, Zgrid)
                cs = plt.contour(
                    RR,
                    ZZ,
                    g["chirz"],
                    levels=[
                        g["chi_axis"] + bdry_dist * (g["chi_boundary"] - g["chi_axis"])
                    ],
                )

                # Now we extract the boundary contour from the contour plot
                v = cs.collections[0].get_paths()[0].vertices
                g["rbbbs"] = v[:, 0]
                g["zbbbs"] = v[:, 1]
                g["nbbbs"] = len(g["rbbbs"])
                plt.close()

        ns = len(g["q"])
        chiN = np.linspace(0, 1, ns)
        chi_scale = g["chi_boundary"] - g["chi_axis"]
        # chi is the poloidal flux
        # psi is the toroidal flux

        # iota = dchi / dpsi = dchi / dchiN * dchiN / dpsi = dchiN / dpsi * (chi_scale)
        # dpsi = 1/iota dchi = q dchiN
        # psi = integral(q dchiN) * (chi_scale)

        # q * (chi(1) - chi(0)) as function of chiN
        q_spline = CubicSpline(chiN, g["q"] * chi_scale)
        # this gives psi(chiN)
        psi_spline = q_spline.antiderivative()

        psi = psi_spline(chiN)
        Psi_b = psi[-1]  # boundary flux

        if np.diff(psi)[0] > 0:
            pressure_psi = CubicSpline(psi, g["pres"])
            iota_psi = CubicSpline(psi, 1 / g["q"])
        else:  # monotonic deacreasing
            pressure_psi = CubicSpline(psi[::-1], g["pres"][::-1])
            iota_psi = CubicSpline(
                psi[::-1], 1 / g["q"][::-1]
            )  # tokmaks only right nowle(1 / g["q"], psi)

        # Defining splines and recalculating important quantities if
        # bdry_dist less than 1. Also works for bdry_dist equal to 1.
        # For bdry_dist equal 1, all quantities *_truncated are the
        # same as quantities without _truncated
        chiN_truncated = np.linspace(0, bdry_dist, ns)
        q_spline_truncated = CubicSpline(chiN_truncated, q_spline(chiN_truncated))

        psi_spline_truncated = q_spline_truncated.antiderivative()
        psi_truncated = psi_spline_truncated(chiN_truncated)
        Psi_b_truncated = psi_truncated[-1]  # boundary flux
        psiN_truncated = psi_truncated / Psi_b_truncated
        rho_truncated = np.sqrt(psiN_truncated)

        pressure = SplineProfile(pressure_psi(psi_truncated), rho_truncated)
        iota = SplineProfile(iota_psi(psi_truncated), rho_truncated)

        nbdry = g["nbbbs"]
        rbdry = g["rbbbs"]
        zbdry = g["zbbbs"]
        theta = np.linspace(0, 2 * np.pi, nbdry)  # rbbbs includes endpoints

        grid = LinearGrid(theta=theta)
        basis = DoubleFourierSeries(M=M, N=N, sym=False)
        transform = Transform(grid, basis, build_pinv=True)
        Rb_lmn = transform.fit(rbdry)
        Zb_lmn = transform.fit(zbdry)

        surf = FourierRZToroidalSurface(
            R_lmn=Rb_lmn,
            Z_lmn=Zb_lmn,
            modes_R=basis.modes[:, 1:],
            modes_Z=basis.modes[:, 1:],
        )

        inputs = {}

        inputs["pressure"] = pressure
        inputs["iota"] = iota
        inputs["sym"] = False
        inputs["surface"] = surf
        inputs["NFP"] = int(1)
        inputs["Psi"] = Psi_b
        inputs["L"] = L
        inputs["M"] = M
        inputs["N"] = N

        eq = Equilibrium(**inputs)

        return eq

    @classmethod
    def efit_to_desc_input(  # noqa: C901 - fxn too complex
        cls,
        eq,
        outfile,
        objective="force",
        optimizer="lsq-exact",
        ftol=1e-2,
        xtol=1e-2,
        gtol=1e-6,
        maxiter=100,
        threshold=1e-8,
    ):
        """Generate a DESC input file from a DESC equilibrium object.

        DESC will automatically choose continuation parameters

        Parameters
        ----------
        eq: Equilibrium
            DESC equilibrium object
        outfile : str or path-like
            name of the DESC input file to create
        objective : str
            objective type used in the input file
        optimizer : str
            type of optimizer
        ftol : float
            relative tolerance of the objective function f
        xtol : float
            relative tolerance of the state vector x
        gtol : float
            absolute tolerance of the projected gradient g
        maxiter : int
            maximum number of optimizer iterations per continuation step
        threshold : float
            Fourier coefficients below this value will be set to 0.
        """
        with open(outfile, "w+") as f:
            header = "# DESC input file generated from an geqdsk file\n# "
            f.write(header + "\n")

            f.write("\n# global parameters\n")
            f.write("sym = {:d}\n".format(eq.sym))
            f.write("NFP = {:d}\n".format(int(eq.NFP)))
            f.write("Psi = {:.8E}\n".format(eq.Psi))

            f.write("\n# spectral resolution\n")
            for key, val in {
                "L_rad": "L",
                "M_pol": "M",
                "N_tor": "N",
                "L_grid": "L_grid",
                "M_grid": "M_grid",
                "N_grid": "N_grid",
            }.items():
                f.write(f"{key} = {getattr(eq, val)}\n")

            f.write("\n# solver tolerances\n")
            f.write("ftol = {:.3E}\n".format(ftol))
            f.write("xtol = {:.3E}\n".format(xtol))
            f.write("gtol = {:.3E}\n".format(gtol))
            f.write("maxiter = {:d}\n".format(maxiter))

            f.write("\n# solver methods\n")
            f.write(f"optimizer = {optimizer}\n")
            f.write(f"objective = {objective}\n")
            f.write("spectral_indexing = {}\n".format(eq._spectral_indexing))

            f.write("\n# pressure and rotational transform/current profiles\n")

            grid = LinearGrid(L=eq.L_grid, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
            rho = grid.nodes[grid._unique_rho_idx, 0]

            pressure = grid.compress(eq.compute("p", grid=grid)["p"])
            iota = grid.compress(eq.compute("iota", grid=grid)["iota"])

            pres_profile = PowerSeriesProfile.from_values(
                rho, pressure, order=eq.L, sym=False
            ).params
            iota_profile = PowerSeriesProfile.from_values(
                rho, iota, order=eq.L, sym=False
            ).params

            if eq.iota:
                char = "i"
                profile = iota_profile

            idxs = np.linspace(0, eq.L - 1, eq.L, dtype=int)
            for l in idxs:
                f.write(
                    "l: {:3d}  p = {:15.8E}  {} = {:15.8E}\n".format(
                        int(l), pres_profile[l], char, profile[l]
                    )
                )

            f.write("\n# fixed-boundary surface shape\n")

            # boundary parameters (eqdsk equilibria are always up-down asymmetric)
            # so we keep all the modes
            for k, (l, m, n) in enumerate(eq.surface.R_basis.modes):
                if abs(eq.Rb_lmn[k]) > threshold or abs(eq.Zb_lmn[k]) > threshold:
                    f.write(
                        "l: {:3d}  m: {:3d}  n: {:3d}  ".format(int(0), m, n)
                        + "R1 = {:15.8E}  Z1 = {:15.8E}\n".format(
                            eq.Rb_lmn[k], eq.Zb_lmn[k]
                        )
                    )

            f.close()
