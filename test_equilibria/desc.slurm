#!/bin/bash
#SBATCH --job-name=desc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=5:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pkim1236@terpmail.umd.edu

#module purge
#module load intel/2021.1.2
#module load intel-mpi/intel/2021.1.1
#module load fftw/intel-2021.1/intel-mpi/3.3.9
#module load netcdf/intel-2021.1/hdf5-1.10.6/intel-mpi/4.7.4
#module load hdf5/intel-2021.1/intel-mpi/1.10.6
#module load gsl/2.6
#module load anaconda3/2021.5


cd /scratch/gpfs/pk2354/DESC/test_equilibria
#srun /home/pk2354/src/stella/stella simple.in
srun python3 /scratch/gpfs/pk2354/DESC/test_constr_qs_con.py
#srun /scratch/gpfs/ethier/STELLAR/STELLA_GK/stella/stella
