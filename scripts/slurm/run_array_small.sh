#!/bin/bash
#SBATCH --job-name=idbench_syn_small
#SBATCH --output=logs/idbench_syn_small_%A_%a.out
#SBATCH --error=logs/idbench_syn_small_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-9

export BASE_SEED=${SLURM_ARRAY_TASK_ID}

# Ensure logs directory exists so Slurm can write output files
mkdir -p ${PWD}/logs

# Source conda initialization explicitly to ensure `conda activate` works
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/local/scratch/cfikes/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/local/scratch/cfikes/miniconda3/etc/profile.d/conda.sh"
fi

set -x
echo "Starting small job on $(hostname) at $(date); workdir=$(pwd); user=$(whoami)"
conda activate cs_534_cuda
# ensure we run from the repository root so `python -m src...` works
cd /local/scratch/cfikes/CS_534_ML_project || exit 1

# remove old figures so small run overwrites them
rm -rf results/figs || true

python -m src.experiments.run_experiments --mode small --workers 1 --base-seed ${BASE_SEED}
