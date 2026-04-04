#!/bin/bash
#SBATCH --job-name=mat-free     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=40G         # memory per cpu-core
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=00:50:00          # total run time limit (HH:MM:SS)
#SBATCH --constraint=nomig        # choose a100 or v100
#SBATCH --mail-type=begin,end,fail  # receive email notifications
#SBATCH --output=newcomb_out.log
#SBATCH --error=newcomb_err.log
#SBATCH --mail-user=ma0566@princeton.edu

module load anaconda3/2024.10
module load cudatoolkit/12.9
conda activate desc-env

python test_gpu.py
python run_newcomb.py 
