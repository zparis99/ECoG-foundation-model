#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --time=15:00:00
#SBATCH -o './logs/%x.out'
#SBATCH -e './logs/%x.err'

module purge
module load anaconda3/2024.2
conda activate /scratch/gpfs/zparis/ECoG-foundation-model/ecog

echo 'Requester:' $USER 'Node:' $HOSTNAME
echo "$@"
echo 'Start time:' `date`
start=$(date +%s)

python "$@"

end=$(date +%s)
echo 'End time:' `date`
echo "Elapsed Time: $(($end-$start)) seconds"
