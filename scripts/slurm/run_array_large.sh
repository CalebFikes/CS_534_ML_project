#!/bin/bash
#SBATCH --job-name=idbench_syn_large
#SBATCH --output=logs/idbench_syn_large_%A_%a.out
#SBATCH --error=logs/idbench_syn_large_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --mem=24G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --array=0-2

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
echo "Starting large worker on $(hostname) at $(date); workdir=$(pwd); user=$(whoami); task=${SLURM_ARRAY_TASK_ID}"
conda activate cs_534_gpu
# ensure we run from the repository root so `python -m src...` works
cd /local/scratch/cfikes/CS_534_ML_project || exit 1

# Run the experiment driver in large mode (worker only)
python -m src.experiments.run_experiments --mode large --workers ${SLURM_CPUS_PER_TASK:-1} --base-seed ${BASE_SEED:-0}
