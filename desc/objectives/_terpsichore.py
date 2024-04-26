import numpy as np
import subprocess
from scipy.interpolate import interp1d
import os
import time
import math
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
    def __init__(self, eq=None, target=0, weight=1, grid=None, name="TERPSICHORE", path=None, bounds=None,normalize=False, submit_script_name="terps_job.submit", normalize_target=False, awall=1.3, deltajp=5.e-1, modelk=0, al0=-5.e-2, nev=1, nfp=2, xplo=1.e-6, max_bozm=19, max_bozn=14, mode_family=0, max_modem=55, min_moden=-8, max_moden=11):
        
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
        self.min_moden = min_moden
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


        wout_filename = "wout_C640_MPOL-7_NTOR-6_NS-32.nc"
        self.path = path
        self.submit_script_name = submit_script_name
        self.submit_script_path = os.path.join(self.path, self.submit_script_name)
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

        self._eq_keys = [
            "iota",
            "iota_r",
            "a",
            "rho",
            "psi",
        ]
        '''
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
        '''
        if verbose > 0:
            print("Precomputing transforms")
        #timer.start("Precomputing transforms")

        '''
        #Need separate transforms and profiles for the equilibrium and flux-tube
        self.eq = eq
        #self._profiles = get_profiles(self._field_line_keys, obj=eq, grid=self.grid)
        self._profiles_eq = get_profiles(self._eq_keys, obj=eq, grid=self.grid)
        #self._transforms = get_transforms(self._field_line_keys, obj=eq, grid=self.grid)
        self._transforms_eq = get_transforms(self._eq_keys, obj=eq, grid=self.grid)

        self._constants = {
            "transforms": self._transforms_eq,
            "profiles": self._profiles_eq,
        }


        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

#        self._check_dimensions()
#        self._set_dimensions(eq)
        '''
        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):

        return self.terps_compute.bind(params,constants)
        
    def compute_impl(self, params, constants):

#        params, constants = self._parse_args(*args, **kwargs)
        if not np.any(params["R_lmn"]):
            return 0

        if constants is None:
            constants = self.constants

        #self.write_vmec()   # Write VMEC file from DESC
        self.compute_fort18()
        self.write_terps_io()
        self.run_terps()
        self.terps_outfile = os.path.join(self.path,'fort.16') # Let's change the name of this at some point
        self.parse_terps_outfile()

        print("Growth rate = {}".format(self.growth_rate))
        
        return jnp.atleast_1d(self.growth_rate)
        

    def write_vmec(self):

        print("Figure out how to do this directly from DESC equilibrium quantities!!")
        
        #VMECIO.save(eq, "path/to/wout.nc", surfs=32)
        
    
    def compute_fort18(self):

        print("Figure out how to do this directly from DESC equilibrium quantities!!")
        
        fs = open('stdout.vmec2terps','w')
        head, tail = os.path.split(self.wout_file)
        cmd = [self.vmec2terps_app, tail]
        subprocess.run(cmd,stdout=fs)
        fs.close()

    def is_terps_complete(self, slurm_file, stop_time, running, runtime):

        if not os.path.exists(slurm_file):
            return False
        else:
            f_slurm = open(slurm_file, 'r')

        if (running):
            print("Stop time = {} seconds".format(stop_time))
            print("Current runtime = {} seconds".format(math.ceil(runtime)))
            if (runtime > stop_time):
                print("TERPS was unable to find a growth rate. Exiting...")
                exit()
            
        terps_out_contents = f_slurm.read()

        if 'GROWTH RATE' in terps_out_contents:
            f_slurm.close()
            rm_cmd = ['rm', 'tpr16_dat_wall'] # There's probably a better way to handle this
            subprocess.run(rm_cmd)
            return True
        else:
            f_slurm.close()
            return False
        
    def run_terps(self):

        sleep_time = 10 # seconds
        stop_time = 300 # seconds (kill the infinite loop if TERPS ran into an error and won't be printing growth rate)
        
        fs = open('stdout.terps','w')
        head, tail = os.path.split(self.terps_infile)
        
        # This could potentially launch a number of parallel TERPS jobs (probably at least want to run parallel jobs for N=0 and N=1 family)
        
        if not (os.path.exists(self.submit_script_path)):
            f = open(self.submit_script_path,"w")
            f.write("#!/bin/bash\n")
            f.write("#SBATCH --job-name=terps    # Job name\n")
            f.write("#SBATCH --time=00:45:00               # Time limit hrs:min:sec\n")
            f.write("#SBATCH --output=terps_%j.log   # Standard output and error log\n")
            f.write("#SBATCH --nodes=1\n")
            f.write("#SBATCH --mem-per-cpu=100G\n")
            f.write("#SBATCH --ntasks-per-node=1\n")
            f.write("#SBATCH --partition=stellar-debug\n")
            f.write("\n")
            f.write("srun {} < {}\n".format(self.terps_app,tail))
            f.close()

        print("Need a command to remove remnants of pasts TERPS runs (tpr16_dat_wall in particular) or move them to a new directory")

        if (os.path.exists(os.path.join(self.path, 'tpr16_dat_wall'))):
            rm_cmd = ['rm', 'tpr16_dat_wall'] # There's probably a better way to handle this
            subprocess.run(rm_cmd)
            
        cmd = ['sbatch', self.submit_script_path]
        terps_process = subprocess.run(cmd,stdout=subprocess.PIPE)
        out_text = terps_process.stdout.decode('utf-8')
        fs.write(out_text)
        jobID = out_text.split()[-1]
        slurm_file = os.path.join(self.path,"terps_{}.log".format(jobID))

        running = False
        runtime = 0.0
        tic = time.perf_counter()
        while not self.is_terps_complete(slurm_file, stop_time, running, runtime):
            if not running:
                check_status_cmd = ['squeue', '-j', jobID, '--format="%T"']
                check_status = subprocess.run(check_status_cmd, stdout=subprocess.PIPE)
                status_text = check_status.stdout.decode('utf-8')
                status = status_text.split()[1].replace('"','')
                if status == 'RUNNING':
                    running = True
                    tic = time.perf_counter()
                    print("TERPS has started running")
                elif status == 'PENDING':
                    print("TERPS is still in the queue")
                else:
                    print(status)
                    exit()
                    
            else:
                print("Growth rate not found. Checking again in {} seconds".format(sleep_time))

            time.sleep(sleep_time)
            toc = time.perf_counter()
            runtime = toc-tic

        print("Found growth rate!")
        
        fs.close()

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

        self.terps_infile = os.path.join(self.path, "{}_N{}_family".format(eq_identifier, self.mode_family))
        f = open(self.terps_infile,"w")

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
        #for _im in range(1,self.max_bozm+1):
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
