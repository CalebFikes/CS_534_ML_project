#!/bin/bash
#SBATCH --job-name=repair_smae
#SBATCH --output=logs/repair_masked_ae_%A_%a.out
#SBATCH --error=logs/repair_masked_ae_%A_%a.err
#SBATCH --array=0-3
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# Map array tasks to CSVs
CSVS=(results/synthetic_large.task0.csv results/synthetic_large.task1.csv results/mnist_large.task0.csv results/mnist_large.task1.csv)
OUTS=(results/synthetic_large.task0.repaired.csv results/synthetic_large.task1.repaired.csv results/mnist_large.task0.repaired.csv results/mnist_large.task1.repaired.csv)

CSV=${CSVS[$SLURM_ARRAY_TASK_ID]}
OUT=${OUTS[$SLURM_ARRAY_TASK_ID]}

echo "Running repair on $CSV -> $OUT"
cd $(dirname "$0")/../../ || exit 1
# ensure we run from repo root
REPO_ROOT=$(pwd)
export PYTHONPATH=$REPO_ROOT

# Use GPU-enabled conda environment python executable
PYTHON=/local/scratch/cfikes/miniconda3/envs/cs_534_gpu/bin/python

$PYTHON scripts/experiments/repair_masked_ae.py --csv "$CSV" --out "$OUT" --max-samples 5000
