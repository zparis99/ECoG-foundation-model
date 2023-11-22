#!/bin/bash
#SBATCH --account=medarc
#SBATCH --partition=h80
#SBATCH --job-name=openneuro
#SBATCH --nodes=1              
#SBATCH --ntasks-per-node=1     # should = number of gpus
#SBATCH --gres=gpu:1
#SBATCH --time=99:00:00        # total run time limit (HH:MM:SS)
#SBATCH -e slurms/%j.err        # first create a "slurms" folder in current directory to store logs
#SBATCH -o slurms/%j.out
#SBATCH --comment=medarc

source ~/.bashrc

cd /fsx/proj-fmri/paulscotti/fMRI-foundation-model/dataset_creation

jupyter nbconvert webdataset_creation_no_afni.ipynb --to python

srun python -u webdataset_creation_no_afni.py