#!/bin/bash

#SBATCH --constraint=gpu80
#SBATCH --nodes=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00


module purge;\

module load anaconda3/2023.3;\

conda activate desc-env;\


python3 -u regen_equilibria.py


