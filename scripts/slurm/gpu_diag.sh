#!/bin/bash
#SBATCH --job-name=check_gpu_env
#SBATCH --output=logs/check_gpu_env_%j.out
#SBATCH --error=logs/check_gpu_env_%j.err
#SBATCH --time=00:05:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

# minimal diagnostic: activate env and run small checks
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/local/scratch/cfikes/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/local/scratch/cfikes/miniconda3/etc/profile.d/conda.sh"
fi
conda activate cs_534_cuda

echo "HOST: $(hostname)"
echo "DATE: $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
python - <<'PY'
import os
print('python:', os.environ.get('PYTHONPATH', ''))
try:
    import torch
    print('torch_version', torch.__version__)
    print('torch.cuda.is_available()', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('cuda device count', torch.cuda.device_count())
except Exception as e:
    print('torch import error', e)
try:
    import faiss
    print('faiss import ok')
    try:
        import numpy as np
        x = np.random.rand(512, 16).astype('float32')
        res = faiss.StandardGpuResources()
        idx = faiss.IndexFlatL2(16)
        gpu_idx = faiss.index_cpu_to_gpu(res, 0, idx)
        gpu_idx.add(x)
        D, I = gpu_idx.search(x[:5], 5)
        print('faiss gpu search shape', D.shape)
    except Exception as e:
        print('faiss gpu test failed', e)
except Exception as e:
    print('faiss import failed', e)
print('ENV CUDA_VISIBLE_DEVICES=', os.environ.get('CUDA_VISIBLE_DEVICES'))
PY

# also show nvidia-smi snapshot
nvidia-smi
