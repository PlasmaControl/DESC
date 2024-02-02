#!/bin/bash
#SBATCH --job-name=DPS
#SBATCH --nodes=1
#SBATCH -n 1
#SBATCH --mem=16G
#SBATCH --time=1:30:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --mail-type=ALL
#SBATCH --mail-user=joaopedrofbiu@tecnico.ulisboa.pt

module purge
module load anaconda
conda activate desc-env

python DPS.py
