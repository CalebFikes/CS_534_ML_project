# Intrinsic Dimension Benchmark

Setup (recommended using conda/miniconda):

CPU (minimal):

```bash
conda create -n id-bench python=3.10 -y
conda activate id-bench
pip install -r requirements.txt
# scikit-dimension may need to be installed separately if not available via pip:
# pip install scikit-dimension

GPU / CUDA notes
-----------------
This repo separates CPU vs GPU install steps. For the GPU cluster (CUDA 12.7) do NOT rely on the plain `requirements.txt` to install PyTorch.

Recommended flow for the cluster (conda/miniconda available):

```bash
conda create -n cs_534_final python=3.10 -y
conda activate cs_534_final
# install binary numba/llvmlite via conda to avoid source builds
conda install -y -c conda-forge llvmlite numba matplotlib
# install remaining Python deps (CPU-only packages)
pip install -r requirements_CUDA.txt
# install PyTorch/FAISS compiled against the cluster CUDA runtime following https://pytorch.org
# Example placeholder (adjust CUDA version per pytorch.org):
# Install PyTorch for the cluster CUDA runtime:
conda install pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia
# Install FAISS via conda (pinned to a version compatible with numpy==1.24.4):
# CPU (conda-forge):
#   conda install -c conda-forge faiss-cpu=1.7.3
# GPU (pytorch + nvidia channels):
#   conda install -c pytorch -c nvidia faiss-gpu=1.7.3
# Note: prefer conda installs for FAISS to ensure the binary is built against
# the same NumPy ABI as the rest of the environment. Installing FAISS via
# pip may pull a wheel built against a different NumPy ABI (e.g. NumPy>=1.25)
# and cause runtime errors. Use the conda commands above on the cluster.
```

If you want to run on CPU only locally, use `requirements.txt` (this repo pins `numpy==1.24.4`) to avoid binary incompatibilities with compiled extensions.

Notes on estimators and GPU
---------------------------
- The estimators implemented here (Levina--Bickel, TwoNN, CorrInt, DANCo, MiND, FisherS via `skdim`) are CPU-based. `skdim` and `scikit-learn` use NumPy/Scipy/numba and do not automatically run on CUDA.
- For large-scale experiments on the cluster we will parallelize across Monte Carlo replicates and reuse precomputed nearest-neighbor structures.
- If you want GPU-accelerated nearest-neighbor computations, I recommend integrating FAISS (faiss-gpu) or using PyTorch/Cupy implementations for pairwise distances. I can add optional hooks for FAISS later.

Quick collaborator notes
------------------------
- **Dependencies:** `requirements.txt` already lists `scikit-dimension`; use `requirements_CUDA.txt` for cluster setup and follow the `conda`/`faiss-gpu` notes there.
- **MNIST / AE:** the driver trains AEs on the full MNIST subset (60000 examples) by default. Expect one AE training to take ~30s on an A6000 when using CUDA (25 epochs in current config).
- **Estimators runtime:** some estimators (notably `corrint` and parts of `fisher`) are very slow on full MNIST latents (per-k runtimes measured in hundreds-to-thousands of seconds). Consider:
	- running MNIST estimator jobs as separate GPU/CPU jobs per seed or bottleneck (use the Slurm templates in `scripts/slurm/`), or
	- excluding `corrint` from large full-array runs and computing it selectively for a subset of seeds, or
	- precomputing and reusing latents in `data/mnist_latents_*.npy` to avoid retraining.
- **CSV outputs:** per-task CSVs are written under `results/` (synthetic CSVs are present already). If MNIST per-task CSVs are missing, the MNIST estimator stage may still be running or was interrupted; check `logs/idbench_syn_small_*.out` for per-estimator timing markers like `[MNIST EST DONE]`.

```

GPU (CUDA 12.7, example using conda):

```bash
conda create -n id-bench python=3.10 -y
conda activate id-bench
# Install pytorch for CUDA 12.7 - pick the matching wheel from pytorch.org if necessary
pip install -r requirements.txt
```

Quick commands:

Run a tiny smoke experiment (synthetic + MNIST subset):

```bash
python -m src.experiments.run_experiments --mode smoke
```

Or run synthetic-only quick runner (no MNIST training):

```bash
python scripts/run_synth.py
```

Cluster / Slurm usage
---------------------

Example Slurm job-array templates are provided in `scripts/slurm/`:

- `run_array.sh`: synthetic job-array template that sets `BASE_SEED` from `SLURM_ARRAY_TASK_ID` and runs the experiment driver (use `--mode small` or `--mode final` as needed).
- `run_mnist_array.sh`: MNIST training template for GPU nodes. It sets `BASE_SEED` from the array index and runs the driver in `--mode smoke`.

On the cluster, create the conda env from `environment.yml`, then install PyTorch/FAISS via conda as noted in `environment.yml` / `requirements_CUDA.txt`. Submit the job-array like:

```bash
sbatch --array=0-9 scripts/slurm/run_array.sh
sbatch --array=0-3 scripts/slurm/run_mnist_array.sh
```

The job templates rely on `BASE_SEED` to spread Monte Carlo replicates across array tasks. The driver will pick up the env var automatically.

Regenerating results on the cluster
----------------------------------

All analysis plots are generated by running the experiment driver (it will save CSVs and plots under `results/`). If you need to regenerate synthetic plots only (no MNIST training), use:

```bash
python scripts/run_synth.py
```

If you need to retrain MNIST models or regenerate full smoke outputs, run the driver:

```bash
python -m src.experiments.run_experiments --mode smoke
```

Git and data
------------

We added `.gitignore` to avoid committing large artifacts (plots, CSVs, latents, saved models). The source, scripts, and configs needed to reproduce the results are kept under version control. To reproduce data/artifacts on the cluster, follow the environment setup and run the scripts above — they will recreate `results/`, `models/`, and `data/` files.

