#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --time=2-00:00:00
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=map_gpu:0,1,2,3
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pkim1236@terpmail.umd.edu
#SBATCH --mem=32G

#module load cudatoolkit 
#module load openmpi/gcc/4.0.4/64
#module load netcdf/gcc/hdf5-1.10.6/openmpi-4.0.4/4.7.3
#module load hdf5/gcc/openmpi-4.0.4/1.10.6

module load nvhpc-nocompiler/22.5
module load openmpi/cuda-11.0/gcc/4.0.4/64
module load netcdf/gcc/hdf5-1.10.6/openmpi-4.0.4/4.7.3
module load hdf5/gcc/openmpi-4.0.4/1.10.6
module load gsl/1.15
module load anaconda3/2021.5
module load fftw/gcc/openmpi-4.0.4/3.3.8


#export JAX_LOG_COMPILES=1
cd /scratch/gpfs/pk2354/DESC/GX
python3 /scratch/gpfs/pk2354/DESC/precise_qh_gx_split_stochastic_swap.py
#python3 /scratch/gpfs/pk2354/DESC/precise_qh_gx_split_stochastic_asp.py

#python3 /scratch/gpfs/pk2354/DESC/test_gxwrapper.py
#python3 /scratch/gpfs/pk2354/DESC/scan_alpha.py
#python3 /scratch/gpfs/pk2354/DESC/gx_scan.py
#python3 /scratch/gpfs/pk2354/DESC/test_gx.py
#cd /scratch/gpfs/pk2354/DESC
#python -m desc /scratch/gpfs/pk2354/DESC/examples/VMEC/input.set_of_configs_for_gx_desc





