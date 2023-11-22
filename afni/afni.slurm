#!/bin/bash
#SBATCH --account=medarc
#SBATCH --partition=g40
#SBATCH --job-name=openneuro
#SBATCH --nodes=1              
#SBATCH --ntasks-per-node=1     # should = number of gpus
#SBATCH --gres=gpu:1
#SBATCH --time=200:00:00          # total run time limit (HH:MM:SS)
#SBATCH -e slurms/%j.err        # first create a "slurms" folder in current directory to store logs
#SBATCH -o slurms/%j.out
#SBATCH --comment=medarc

source /admin/home-paulscotti/.bashrc

cd /fsx/proj-fmri/paulscotti/fMRI-foundation-model/afni

srun python -u no_afni.py