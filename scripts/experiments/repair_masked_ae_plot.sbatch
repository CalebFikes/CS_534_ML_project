#!/bin/bash
#SBATCH --job-name=repair_plot
#SBATCH --output=logs/repair_plot_%j.out
#SBATCH --error=logs/repair_plot_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

cd /local/scratch/cfikes/CS_534_ML_project
export PYTHONPATH="$PWD"
PYTHON=/local/scratch/cfikes/miniconda3/envs/cs_534_gpu/bin/python

echo "Running plotting: $PYTHON scripts/plot_results.py --size large"
$PYTHON scripts/plot_results.py --size large

echo "Plotting finished at $(date)"
