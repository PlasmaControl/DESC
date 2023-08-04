#!/bin/bash
#SBATCH --job-name=convtest_jc      # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --mem=64G
#SBATCH --time=01:02:00          # total run time limit (HH:MM:SS)
#SBATCH --array=1-80            
#SBATCH --mail-type=all          # send email on job start, end and fault
#SBATCH --mail-user=jc9880@princeton.edu

echo "My SLURM_ARRAY_JOB_ID is $SLURM_ARRAY_JOB_ID."
echo "My SLURM_ARRAY_TASK_ID is $SLURM_ARRAY_TASK_ID"
echo "Executing on the machine:" $jaxcrocker

module purge
module load anaconda3/2023.3
conda activate desc-env

python GamCRunner.py