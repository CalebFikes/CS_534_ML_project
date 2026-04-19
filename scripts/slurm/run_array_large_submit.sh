#!/bin/bash
# Submission wrapper for large array: deletes old outputs, submits array workers, then schedules finalize job.
set -euo pipefail
ROOT=/local/scratch/cfikes/CS_534_ML_project
LOGS=${ROOT}/logs
mkdir -p "$LOGS"

echo "Cleaning old results and figures for large"
rm -f ${ROOT}/results/*.csv || true
rm -f ${ROOT}/results/*.npz || true
rm -rf ${ROOT}/results/figs_large || true

echo "Submitting large array workers"
array_jobid=$(sbatch --parsable ${ROOT}/scripts/slurm/run_array_large.sh)
echo "Submitted array job id ${array_jobid}"

echo "Submitting finalize job dependent on array completion"
sbatch --dependency=afterok:${array_jobid} --job-name=idbench_syn_large_finalize \
  --output=${ROOT}/logs/idbench_syn_large_finalize_%j.out \
  --error=${ROOT}/logs/idbench_syn_large_finalize_%j.err \
  --wrap="source \"$HOME/miniconda3/etc/profile.d/conda.sh\" || true; conda activate cs_534_gpu || true; cd ${ROOT} && PYTHONPATH=. python scripts/plot_results.py --size large --finalize"

echo "Done. Array job ${array_jobid} and finalize submitted."
