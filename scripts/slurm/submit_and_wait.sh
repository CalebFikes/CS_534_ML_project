#!/usr/bin/env bash
# Submit an sbatch array and wait for it to finish, then aggregate and regenerate plots.
# Usage: ./submit_and_wait.sh small

MODE=${1:-small}
SCRIPT="scripts/slurm/run_array_${MODE}.sh"
if [ ! -f "$SCRIPT" ]; then
    echo "Runner script $SCRIPT not found" >&2
    exit 1
fi

echo "Submitting $SCRIPT"
OUT=$(sbatch "$SCRIPT")
echo "$OUT"
JOBID=$(echo "$OUT" | awk '{print $4}')
if [ -z "$JOBID" ]; then
    echo "Failed to parse job id from sbatch output" >&2
    exit 1
fi

echo "Submitted batch job $JOBID; waiting for completion..."
while squeue -j $JOBID -h | grep -q .; do
    sleep 10
done
echo "Job $JOBID finished at $(date)"

# Activate environment and aggregate
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/local/scratch/cfikes/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/local/scratch/cfikes/miniconda3/etc/profile.d/conda.sh"
fi
conda activate cs_534_cuda
cd /local/scratch/cfikes/CS_534_ML_project || exit 1

echo "Aggregating per-task CSVs for mode=$MODE"
python scripts/aggregate_results.py results/${MODE}_*.task*.csv --out-raw results/${MODE}_combined.csv --out-summary results/${MODE}_summary.csv || echo "Aggregation failed or no per-task CSVs found"

echo "Regenerating plots"
PYTHONPATH=. python scripts/regenerate_plots.py || echo "Plot regen failed"
echo "Done at $(date)"
