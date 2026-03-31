#!/bin/bash
#SBATCH --job-name=idbench_syn_cuda
#SBATCH --output=logs/idbench_syn_cuda_%A_%a.out
#SBATCH --error=logs/idbench_syn_cuda_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-9

export BASE_SEED=${SLURM_ARRAY_TASK_ID}

module load anaconda3
conda activate cs_534_cuda

# Run the experiment driver in small mode
python -m src.experiments.run_experiments --mode small --workers 1 --base-seed ${BASE_SEED}
