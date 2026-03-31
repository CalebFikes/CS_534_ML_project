#!/bin/bash
#SBATCH --job-name=idbench_mnist
#SBATCH --output=logs/idbench_mnist_%A_%a.out
#SBATCH --error=logs/idbench_mnist_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-3

# Each array task will run the MNIST autoencoder smoke with a different base seed
export BASE_SEED=${SLURM_ARRAY_TASK_ID}

module load anaconda3
conda activate cs_534_final

# Run the experiment driver which includes MNIST training in smoke mode
python -m src.experiments.run_experiments --mode smoke --workers 1
