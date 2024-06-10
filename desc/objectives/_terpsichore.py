import numpy as np
import subprocess
from scipy.interpolate import interp1d
import os
import time
import math
import multiprocessing
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
=======

    """

    # Used by GX objective, not sure where they're used yet
    _scalar = True
    _linear = False
    _units = "[1/s]"
    _print_value_fmt = "Growth rate: {:10.3e} "


    # Need to sure up the paths
    def __init__(self, eq=None, target=0, weight=1, grid=None, name="TERPSICHORE", wout_filename="wout_default.nc", try_parallel=True, path=None, bounds=None,normalize=False, submit_script_name="terps_job.submit", normalize_target=False, awall=2.0, deltajp=1.e-2, modelk=1, al0=-5.e-1, nev=1, nfp=2, xplo=1.e-6, max_bozm=19, max_bozn=14, mode_family=0, max_modem=55, min_moden=-8, max_moden=11, nsurf=128):
        
        if target is None and bounds is None:
            target = 0
        self.eq = eq
        self.try_parallel = try_parallel
        self.nsurf = nsurf # No functionality unless using dynamic allocation TERPS
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
        self.min_moden = min_moden
        self.max_moden = max_moden
        # lssl, lssd, lssl_repeat, and lssd_repeat are only used with dynamic allocation TERPS
        self.lssl = 200 # LSSL and LSSD depend on the specified resolution and the required result is given after running the code (once for each variable)
        self.lssd = 100
        self.lssl_repeat = True
        self.lssd_repeat = True
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
        self.terps_stdout = os.path.join(self.path,'stdout.terps')
        self.submit_script_name = submit_script_name
        self.submit_script_path = os.path.join(self.path, self.submit_script_name)
        self.wout_filename = wout_filename
        self.wout_file = os.path.join(self.path, wout_filename)
        self.vmec2terps_app = os.path.join(self.path, "thea-vmec2terps.x")
        self.terps_app = os.path.join(self.path, "tpr_ap.x")

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
        self._dim_f = 1

        timer = Timer()

        super().build(use_jit=use_jit, verbose=verbose)
        

    def compute(self, params, constants=None):

        return self.terps_compute.bind(params,constants)

    
    def compute_impl(self, params, constants):

#        params, constants = self._parse_args(*args, **kwargs)
        if not np.any(params["R_lmn"]):
            return 0

        if constants is None:
            constants = self.constants

        self.write_vmec()   # Write VMEC file from DESC equilibrium
        self.write_terps_io()
        self.run_terps()
        self.terps_outfile = os.path.join(self.path,'fort.16') # Let's change the name of this at some point
        self.parse_terps_outfile()

        print("Growth rate = {}".format(self.growth_rate))
        
        return jnp.atleast_1d(self.growth_rate)
        

    def write_vmec(self):

        from desc.vmec import VMECIO
        VMECIO.save(self.eq, self.wout_file, surfs=self.nsurf)

    def remove_terps_files(self):

        # Keeping this function for now in case there's a reason we might want these files in the future
        
        if (os.path.exists(os.path.join(self.path, 'tpr16_dat_wall'))):
            rm_cmd = ['rm', 'tpr16_dat_wall']
            subprocess.run(rm_cmd)

        if (os.path.exists(os.path.join(self.path, 'tpr16_dat_pvi'))):
            rm_cmd = ['rm', 'tpr16_dat_pvi']
            subprocess.run(rm_cmd)
        

    def is_terps_complete(self, stop_time, runtime, terps_subprocess):

        if (os.path.exists(self.terps_stdout)):
            f_slurm = open(self.terps_stdout, 'r')
        else:
            return False

        print("Current runtime = {} seconds".format(math.ceil(runtime)))
        if (runtime > stop_time):
            print("TERPS was unable to find a growth rate. Exiting...")
            exit()
            
        terps_out_contents = f_slurm.read()
        
        if 'GROWTH RATE' in terps_out_contents:
            if terps_subprocess.returncode == None:
                terps_subprocess.terminate()
            f_slurm.close()
            #self.remove_terps_files()
            return True
        
        elif 'PARAMETER LSSL' in terps_out_contents:
            if terps_subprocess.returncode == None:
                terps_subprocess.terminate()
            f_slurm.seek(0)
            line = f_slurm.readline()
            while line:
                if 'PARAMETER LSSL' in line:
                    self.lssl = int(line.split("TO:")[1])
                    break
                line = f_slurm.readline()
            self.write_terps_io() # Rewrite the input file with suggested value of LSSL and re-run
            return True
        
        elif 'PARAMETER LSSD' in terps_out_contents:
            if terps_subprocess.returncode == None:
                terps_subprocess.terminate()
            f_slurm.seek(0)
            line = f_slurm.readline()
            while line:
                if 'PARAMETER LSSD' in line:
                    self.lssd = int(line.split("TO:")[1])
                    break
                line = f_slurm.readline()
            self.write_terps_io() # Rewrite the input file with suggested value of LSSD and re-run
            return True

        else:
            f_slurm.close()
            return False


    def run_terps(self):
        
        sleep_time = 1 # seconds
        stop_time = 60 # seconds (kill the infinite loop if TERPS ran into an error and won't be printing growth rate)
        
        fs_error = open('error.terps','w')
        fs = open('stdout.terps','w')
        head, tail = os.path.split(self.terps_infile)

        # self.remove_terps_files() # Newest TERPS version does not write out problematic files
        
        cmd = ['srun', '-n', '1', '-t', '00:01:00', '<', self.terps_app, self.wout_filename]
        terps_subprocess = subprocess.run(cmd, stdin=open(self.terps_infile,'r'), stdout=fs, stderr=fs_error)

        runtime = 0.0
        tic = time.perf_counter()
        while not self.is_terps_complete(stop_time, runtime, terps_subprocess):
            time.sleep(sleep_time)
            toc = time.perf_counter()
            runtime = toc-tic

        # !!!!! This check and setting of LSSL and LSSD should be done in a TERPS run BEFORE sending out a bunch of parallel TERPS runs (since they should all use the same LSSL and LSSD values) !!!!!
        if (self.lssl_repeat):
            self.lssl_repeat = False
            self.run_terps()
            
        elif (self.lssd_repeat):
            self.lssd_repeat = False
            self.run_terps()
                    
        fs.close()

        # The parallel jobs could start here

        # Need a command here to wait until all TERPS runs are complete if doing some form of parallel execution
        
        
    def parse_terps_outfile(self):

        # Read fort.16 and search for growth rate
        if (os.path.exists(self.terps_outfile)):
            f = open(self.terps_outfile, 'r')
        else:
            print("TERPS fort.16 output file not found! Exiting...")
            exit()
            
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

    
    def write_terps_io(self):
        
        eq_identifier = "C640"

        ivac = self.nsurf // 4
        if (self.max_moden > 8) or (self.max_modem > 16):
            nj = 150
            nk = 150
        elif (self.max_moden > 4) or (self.max_modem > 8):
            nj = 100
            nk = 100
        else:
            nj = 50
            nk = 50
 
        self.terps_infile = os.path.join(self.path, "{}_N{}_family".format(eq_identifier, self.mode_family))
        f = open(self.terps_infile,"w")

        f.write("               {}\n".format(eq_identifier))
        f.write("C\n")
        f.write("C        MM  NMIN  NMAX   MMS NSMIN NSMAX NPROCS INSOL\n")
        f.write("         {:>2d}   {:>3d}   {:>3d}    55    -8    10    1     0\n".format(self.max_bozm, -self.max_bozn, self.max_bozn))
        f.write("C\n")
        f.write("C        NJ    NK  IVAC  LSSL  LSSD MMAXDF NMAXDF\n")
        f.write("        {:>3d}   {:>3d}   {:>3d}  {:>4d}  {:>4d}    120     64\n".format(nj, nk, ivac, self.lssl, self.lssd))
        f.write("C     TABLE OF FOURIER COEFFIENTS FOR BOOZER COORDINATES\n")
        f.write("C     EQUILIBRIUM SETTINGS ARE COMPUTED FROM FIT/VMEC\n")
        f.write("C\n")

        boz_str_title = "C M=  0"
        boz_str_neg = "      0"
        boz_str_pos = "      1"
        for _im in range(1,37):
            if (_im >= 10):
                boz_str_title += " "+str(_im)[1]
            else:
                boz_str_title += " "+str(_im)
            boz_str_neg += " 1"
            boz_str_pos += " 1"
                

        boz_str_title += "  N\n"
        f.write(boz_str_title)
        for _in in range(-self.max_bozn,self.max_bozn+1):
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
        f.write("  {:10.4e}  1.5000e+00 -1.0000e-00  5.2000e-00 -0.0000e-00 +0.0000e-00      2\n".format(self.awall))
        f.write("C\n")
        f.write("C    RPLMIN       XPLO      DELTAJP       WCT        CURFAC\n")
        f.write("  1.0000e-05  {:10.4e}  {:10.4e}  6.6667e-01  1.0000e-00\n".format(self.xplo, self.deltajp))
        f.write("C\n")
        f.write("C                                                             MODELK =      {}\n".format(self.modelk))
        f.write("C\n")
        f.write("C     NUMBER OF EQUILIBRIUM FIELD PERIODS PER STABILITY PERIOD: NSTA =      {}\n".format(self.nfp))
        f.write("C\n")
        f.write("C     TABLE OF FOURIER COEFFIENTS FOR STABILITY DISPLACEMENTS\n")
        f.write("C\n")
        f.write("C M=  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5  N\n")

        for _in in range(-8,11):

            in_family = False
            if (_in % 2 == self.mode_family): # This needs to be modified for nfp != 2,3
                in_family = True

            if _in <= 0:
                mode_str = "      0"
            else:
                if ((_in <= self.max_moden) and (_in >= self.min_moden) and (in_family)):
                    mode_str = "      1"
                else:
                    mode_str = "      0"
                
            for _im in range(1,55+1):
                if ((_im <= self.max_modem) and (in_family)):
                    if ((_in <= self.max_moden) and (_in >= self.min_moden)):
                        mode_str += " 1"
                    else:
                        mode_str += " 0"
                else:
                    mode_str += " 0"

            mode_str += "{:>3}\n".format(_in)
            f.write(mode_str)

        f.write("C\n")
        f.write("C   NEV NITMAX         AL0     EPSPAM IGREEN MPINIT\n")
        f.write("      {}   4500  {:10.3e}  1.000E-04      0      0\n".format(self.nev,self.al0))
        f.write("C\n")
        
        f.close()
