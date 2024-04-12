import numpy as np
import subprocess
from scipy.interpolate import interp1d
import os
import time
from desc.backend import jnp,put

from .objective_funs import ObjectiveFunction, _Objective
from .utils import (
    factorize_linear_constraints,
)
from desc.utils import Timer
from desc.compute import compute as compute_fun
from desc.compute import get_profiles, get_transforms, get_params
from desc.compute.utils import dot

from scipy.constants import mu_0, pi
from desc.grid import LinearGrid, Grid, ConcentricGrid, QuadratureGrid
from jax import core
from jax.interpreters import ad, batching
from desc.derivatives import FiniteDiffDerivative
import netCDF4 as nc
from shutil import copyfile
from desc.compute.utils import cross, dot


class TERPSICHORE(_Objective):
    r"""Calls the linear MHD stability code TERPSICHORE to compute the linear growth rates of the fastest growing instability

    Parameters
    ----------
    eq : Equilibrium, optional
        Equilibrium that will be optimized to satisfy the Objective.
    target : float, ndarray, optional
         COMPLETE THIS
    bounds : tuple, optional
         COMPLETE THIS
    weight : float, ndarray, optional
         COMPLETE THIS

    more stuff here
    """

    # Used by GX objective, not sure where they're used yet
    _scalar = True
    _linear = False
    _units = "[]" # Outputting with dimensions or normalized?
    _print_value_fmt = "Growth rate: {:10.3e} "


    # Need to sure up the paths
    def __init__(
            self,
            eq=None,
            target=None,
            weight=1,
            grid=None,
            name="TERPSICHORE",
            path=os.getenv['terps_dir'],
            path_in=os.getenv['terps_dir'],
            bounds=None,
            normalize=False,
            normalize_target=False,
            awall=1.5,
            deltajp=4.e-2,
            modelk=0,
            al0=-5.e-2,
            nev=1,
            nfp=2,
            xplo=1.e-6,
            max_bozm=19,
            max_bozn=14,
            mode_family=0,
            max_modem=55,
            max_moden=8,
            wout_filename="" # add something
    ):
        
        if target is None and bounds is None:
            target = 0
        self.eq = eq
        self.awall = awall
        self.deltajp = deltajp
        self.modelk = modelk
        self.al0 = al0
        self.nev = nev
        self.nfp = nfp
        self.xplo = xplo
        self.max_bozm = max_bozm
        self.max_bozn = max_bozn
        self.mode_family = mode_family
        self.max_modem = max_modem
        self.max_moden = max_moden
        self.grid = grid
        
        super().__init__(
            things=eq,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            name=name,
        )

        units = ""
        self._callback_fmt = "Growth rate: {:10.3e} " + units
        self._print_value_fmt = "Growth rate: {:10.3e} " + units
        
        self.path = path
        self.path_in = path_in
        self.wout_file = os.path.join(self.path, wout_filename)
        self.vmec2terps_app = os.path.join(self.path, "thea-vmec2terps.x")
        self.terps_app = os.path.join(self.path, "tpr_ap.x")
        # self.t = t # -- Replace with some stability indicator?

        self.terps_compute = core.Primitive("terps")
        self.terps_compute.def_impl(self.compute_impl)
        # What are the following two lines?
        ad.primitive_jvps[self.terps_compute] = self.compute_terps_jvp
        batching.primitive_batchers[self.terps_compute] = self.compute_terps_batch

    def build(self, use_jit=False, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives
            (MUST be False to run GX).
        verbose : int, optional
            Level of output.

        """
        eq = self.things[0]
        if self.grid is None:
            self.grid_eq = QuadratureGrid(
                L=eq.L_grid,
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=eq.NFP,
            )


        self._dim_f = 1 # Presumbaly this should be 1? just a growth rate
        timer = Timer()

        self._eq_keys = [
            "iota",
            "iota_r",
            "a",
            "rho",
            "psi",
        ]
        self._field_line_keys = [
        "|B|", "|grad(psi)|^2", "grad(|B|)", "grad(alpha)", "grad(psi)",
        "B", "grad(|B|)", "kappa", "B^theta", "B^zeta", "lambda_t", "lambda_z",'p_r',
        "lambda_r", "lambda", "g^rr", "g^rt", "g^rz", "g^tz", "g^tt", "g^zz",
        "e^rho", "e^theta", "e^zeta"
        ]

        self._args = get_params(
            self._field_line_keys,
            obj="desc.equilibrium.equilibrium.Equilibrium",
            has_axis=self.grid_eq.axis.size,
        )

        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")
        
        #Need separate transforms and profiles for the equilibrium and flux-tube
        self.eq = eq
        self._profiles = get_profiles(self._field_line_keys, obj=eq, grid=self.grid)
        self._profiles_eq = get_profiles(self._eq_keys, obj=eq, grid=self.grid_eq)
        self._transforms = get_transforms(self._field_line_keys, obj=eq, grid=self.grid)
        self._transforms_eq = get_transforms(self._eq_keys, obj=eq, grid=self.grid_eq)

        self._constants = {
            "transforms": self._transforms_eq,
            "profiles": self._profiles_eq,
        }

        if self._normalize:
            scales = compute_scaling_factors(eq)
            # I think only needed if we're not using the TERPS normalization?

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

#        self._check_dimensions()
#        self._set_dimensions(eq)
        super().build(use_jit=use_jit, verbose=verbose)

        
    def compute_impl(self, params, constants):
                
#        params, constants = self._parse_args(*args, **kwargs)
        if not np.any(params["R_lmn"]):
            return 0

        if constants is None:
            constants = self.constants

        self.write_vmec()   # Write VMEC file from DESC
        self.compute_fort18()
        self.write_terps_io()
        self.run_terps()

        self.terps_outfile = os.path.join(self.path_in,'fort.16') # Let's change the name of this at some point
        self.parse_terps_outfile()


        # Is something analogous needed for TERPS??
        if qflux_avg > 20:
            out_fail = 'fail_' + str(self.t) + '_.nc'
            copyfile(out_file,out_fail)
            stdout = 'stdout.out_' + str(self.t)
            stdout_fail = 'stdout_fail.out_' + str(self.t)
            copyfile(stdout,stdout_fail)
        return jnp.atleast_1d(qflux_avg)
        

    def write_terps_io(self):
        t = str(self.t) # t is a time indicator here (could change to something stability-relevant
        path_in_old = self.path_in + '.in' # I'm not sure which INPUT FILE changes would be made between calls, wouldn't most be fort.18 related?
        path_in_new = self.path_in + '_' + t + '.in'
        self.write_input(path_in_old,path_geo_old,path_in_new,path_geo_new)


    def compute_fort18(self):

        print("Figure out how to do this directly from DESC equilibrium quantities!!")
        
        stdout = 'stdout.out_' + str(self.t) # Again, replace 't' with something stability-related
        stderr = 'stderr.out_' + str(self.t)
        fs = open('stdout.out_' + str(self.t),'w')
        path_in = self.path_in + "_" + str(self.t) + '.in'
        cmd = [self.vmec2terps_app, self.wout_file]
        subprocess.run(cmd,stdout=fs)
        fs.close()
        # Need to ensure that the output file is in the correct directory
        
    def run_terps(self):
        stdout = 'stdout.out_' + str(self.t) # Again, replace 't' with something stability-related
        stderr = 'stderr.out_' + str(self.t)
        fs = open('stdout.out_' + str(self.t),'w')
        path_in = self.path_in + "_" + str(self.t) + '.in'
        cmd = ['srun', '-N', '1', '-t', '00:45:00', '--ntasks-per-node=1', '--mem-per-cpu=100G', self.terps_app, '<', path_in]
        subprocess.run(cmd,stdout=fs)
        fs.close()

    def parse_terps_outfile(self):

        # Read fort.16 and search for growth rate
        f = open(self.terps_outfile, 'r')
        
        growth_rate = []
        for line_number, line in enumerate(f, start=1):
            index = line.find('GROWTH RATE')
            if index != -1:
                growth_rate.append(float(line.strip().split("=")[1]))

        if (len(growth_rate) > 1):
            print("Currently capable of using only one growth rate. Exiting...")
            exit()
        elif (len(growth_rate) == 0):
            print("Growth rate not found! Exiting...")
            exit()

        self.growth_rate = growth_rate[0]
        
    # I'm not sure what's going on with these final two functions
    def compute_terps_jvp(self,values,tangents):
        
        R_lmn, Z_lmn, L_lmn, i_l, c_l, p_l, Psi = values
        primal_out = jnp.atleast_1d(0.0)

        n = len(values) 
        argnum = np.arange(0,n,1)
        
        jvp = FiniteDiffDerivative.compute_jvp(self.compute,argnum,tangents,*values,rel_step=1e-2)
        
        return (primal_out, jvp)

    def compute_terps_batch(self, values, axis):
        numdiff = len(values[0])
        res = jnp.array([0.0])

        for i in range(numdiff):
            R_lmn = values[0][i]
            Z_lmn = values[1][i]
            L_lmn = values[2][i]
            i_l = values[3][i]
            p_l = values[4][i]
            Psi = values[5][i]
            
            res = jnp.vstack([res,self.compute(R_lmn,Z_lmn,L_lmn,i_l,p_l,Psi)])

        res = res[1:]


        return res, axis[0]

    

    def write_input(self,path_in_temp,geo_temp,path_in,geo):
        copyfile(path_in_temp,path_in) # Need a standard TERPS input

        terps_infile = os.path.join(path_in, "{}_N{}_family".format(eq_identifier, mode_family))
        f = open(terps_infile,"w")

        f.write("               {}\n".format(eq_identifier))
        f.write("C\n")
        f.write("C        MM  NMIN  NMAX   MMS NSMIN NSMAX NPROCS INSOL\n")
        f.write("         {:>2d}   {:>3d}   {:>3d}    55    -8    10    1     0\n".format(self.max_bozm, -self.max_bozn, self.max_bozn))
        f.write("C\n")
        f.write("C     TABLE OF FOURIER COEFFIENTS FOR BOOZER COORDINATES\n")
        f.write("C     EQUILIBRIUM SETTINGS ARE COMPUTED FROM FIT/VMEC\n")
        f.write("C\n")

        boz_str_title = "C M=  0"
        boz_str_neg = "      0"
        boz_str_pos = "      1"
        for _im in range(1,max_bozm+1):
            if (_im >= 10):
                boz_str_title += " "+str(_im)[1]
            else:
                boz_str_title += " "+str(_im)
            boz_str_neg += " 1"
            boz_str_pos += " 1"
                

        boz_str_title += "  N\n"
        f.write(boz_str_title)
        for _in in range(-max_bozn,max_bozn+1):
            final_str_neg = boz_str_neg+"{:>3}\n".format(_in)
            final_str_pos = boz_str_pos+"{:>3}\n".format(_in)
            if _in < 0.0:
                f.write(final_str_neg)
            else:
                f.write(final_str_pos)
            
        f.write("C\n")
        f.write("      LLAMPR      LVMTPR      LMETPR      LFOUPR\n")
        f.write("           0           0           0           0\n")
        f.write("      LLHSPR      LRHSPR      LEIGPR      LEFCPR\n")
        f.write("           9           9           1           1\n")
        f.write("      LXYZPR      LIOTPL      LDW2PL      LEFCPL\n")
        f.write("           0           1           1           1\n")
        f.write("      LCURRF      LMESHP      LMESHV      LITERS\n")
        f.write("           1           1           2           1\n")
        f.write("      LXYZPL      LEFPLS      LEQVPL      LPRESS\n")
        f.write("           1           1           0           0\n")
        f.write("C\n")
        f.write("C    PVAC        PARFAC      QONAX        QN         DSVAC       QVAC    NOWALL\n")
        f.write("  1.0001e+00  0.0000e-00  0.6500e-00  0.0000e-00  1.0000e-00  1.0001e+00     -2\n")
        f.write("\n")
        f.write("C    AWALL       EWALL       DWALL       GWALL       DRWAL       DZWAL   NPWALL\n")
        f.write("  {:10.4e}  1.5000e+00 -1.0000e-00  5.2000e-00 -0.0000e-00 +0.0000e-00      2\n".format(awall))
        f.write("C\n")
        f.write("C    RPLMIN       XPLO      DELTAJP       WCT        CURFAC\n")
        f.write("  1.0000e-05  {:10.4e}  {:10.4e}  6.6667e-01  1.0000e-00\n".format(xplo, deltajp))
        f.write("C\n")
        f.write("C                                                             MODELK =      {}\n".format(modelk))
        f.write("C\n")
        f.write("C     NUMBER OF EQUILIBRIUM FIELD PERIODS PER STABILITY PERIOD: NSTA =      {}\n".format(nfp))
        f.write("C\n")
        f.write("C     TABLE OF FOURIER COEFFIENTS FOR STABILITY DISPLACEMENTS\n")
        f.write("C\n")
        f.write("C M=  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5  N\n")

        #for _in in range(-max_moden,max_moden+1):
        for _in in range(-8,11):

            in_family = False
            if (_in % 2 == mode_family): # This needs to be modified for nfp != 2,3
                in_family = True

            if _in <= 0:
                mode_str = "      0"
            else:
                if ((np.abs(_in) <= max_moden) and (in_family)):
                    mode_str = "      1"
                else:
                    mode_str = "      0"
                
            for _im in range(1,55+1):
                if ((_im <= max_modem) and (in_family)):
                    mode_str += " 1"
                else:
                    mode_str += " 0"

            mode_str += "{:>3}\n".format(_in)
            f.write(mode_str)

      f.write("C\n")
      f.write("C   NEV NITMAX         AL0     EPSPAM IGREEN MPINIT\n")
      f.write("      {}   4500  {:10.3e}  1.000E-04      0      0\n".format(nev,al0))
      f.write("C\n")
        
      f.close()
