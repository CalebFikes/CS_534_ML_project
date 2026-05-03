#!/bin/bash
#SBATCH --job-name=repair_smae_all
#SBATCH --output=logs/repair_masked_ae_all_%A_%a.out
#SBATCH --error=logs/repair_masked_ae_all_%A_%a.err
#SBATCH --array=0-7
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

set -euo pipefail

CSVS=(
  results/synthetic_large.task0.csv
  results/synthetic_large.task1.csv
  results/synthetic_large.task2.csv
  results/synthetic_large.task3.csv
  results/mnist_large.task0.csv
  results/mnist_large.task1.csv
  results/mnist_large.task2.csv
  results/mnist_large.task3.csv
)
OUTS=(
  results/synthetic_large.task0.repaired.csv
  results/synthetic_large.task1.repaired.csv
  results/synthetic_large.task2.repaired.csv
  results/synthetic_large.task3.repaired.csv
  results/mnist_large.task0.repaired.csv
  results/mnist_large.task1.repaired.csv
  results/mnist_large.task2.repaired.csv
  results/mnist_large.task3.repaired.csv
)

IDX=$SLURM_ARRAY_TASK_ID
CSV=${CSVS[$IDX]}
OUT=${OUTS[$IDX]}

echo "Starting repair for ${CSV} (task ${IDX}) at $(date)"

cd /local/scratch/cfikes/CS_534_ML_project
export PYTHONPATH="$PWD"

PYTHON=/local/scratch/cfikes/miniconda3/envs/cs_534_gpu/bin/python

# run and stream output
echo "Running: $PYTHON scripts/experiments/repair_masked_ae.py --csv $CSV --out $OUT --max-samples 0"
$PYTHON scripts/experiments/repair_masked_ae.py --csv "$CSV" --out "$OUT" --max-samples 0

echo "Finished repair for ${CSV} at $(date)"

# Only task 0 schedules a dependent plotting job that will run after the entire array completes
if [ "${SLURM_ARRAY_TASK_ID:-0}" -eq 0 ]; then
  if [ -n "${SLURM_ARRAY_JOB_ID:-}" ]; then
    echo "Task 0 submitting plotting job dependent on array ${SLURM_ARRAY_JOB_ID}"
    sbatch --dependency=afterok:${SLURM_ARRAY_JOB_ID} scripts/experiments/repair_masked_ae_plot.sbatch
  else
    echo "Warning: SLURM_ARRAY_JOB_ID not set; not submitting plotting job"
  fi
fi
