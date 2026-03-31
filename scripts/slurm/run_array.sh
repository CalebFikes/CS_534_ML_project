#!/bin/bash
#SBATCH --job-name=idbench_syn
#SBATCH --output=logs/idbench_syn_%A_%a.out
#SBATCH --error=logs/idbench_syn_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-9

# This Slurm template runs the synthetic experiments with an environment
# variable BASE_SEED set from SLURM_ARRAY_TASK_ID. Each array task will use
# a different base seed so replicates can be parallelized across the array.

export BASE_SEED=${SLURM_ARRAY_TASK_ID}

module load anaconda3
conda activate cs_534_final

# Run the experiment driver (mode can be smoke/small/final as needed)
python -m src.experiments.run_experiments --mode small --workers 1
