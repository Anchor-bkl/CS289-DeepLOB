#!/bin/bash
#SBATCH --job-name=deeplob_repro
#SBATCH --partition=GPU-shared
#SBATCH --account=mth250011p
#SBATCH --gres=gpu:v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=60G
#SBATCH --time=08:00:00
#SBATCH --output=/ocean/projects/mth250011p/xxiao7/DeepLOB/logs/slurm_%j.out
#SBATCH --error=/ocean/projects/mth250011p/xxiao7/DeepLOB/logs/slurm_%j.err

set -e
mkdir -p /ocean/projects/mth250011p/xxiao7/DeepLOB/logs

echo "=== DeepLOB Reproduction Job ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Started:   $(date)"
echo ""

# ---- Environment ----
module purge
module load AI/pytorch_23.02-1.13.1-py3

# All cache/user dirs to ocean (avoid home dir quota)
export PYUSER=/ocean/projects/mth250011p/xxiao7/pyuser
export PYTHONUSERBASE=$PYUSER
export PIP_CACHE_DIR=/ocean/projects/mth250011p/xxiao7/pip_cache
export HF_HOME=/ocean/projects/mth250011p/xxiao7/huggingface-cache
export MPLCONFIGDIR=/ocean/projects/mth250011p/xxiao7/mpl_cache
mkdir -p $MPLCONFIGDIR

# Explicitly add ocean user site-packages to PYTHONPATH so nbconvert kernel sees them
PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
export PYTHONPATH=$PYUSER/lib/python${PYVER}/site-packages:${PYTHONPATH:-}

echo "Python:   $(which python3)"
echo "PYTHONPATH: $PYTHONPATH"
python3 -c "import torch; print('PyTorch:', torch.__version__)"
python3 -c "import torch; cuda=torch.cuda.is_available(); print('CUDA:', cuda, torch.cuda.get_device_name(0) if cuda else '')"
python3 -c "import tqdm, seaborn, torchinfo, statsmodels; print('Extra packages: OK')"
echo ""

# ---- Execute notebook ----
NB_IN=/ocean/projects/mth250011p/xxiao7/DeepLOB/run_deeplob_pytorch.ipynb
NB_OUT=/ocean/projects/mth250011p/xxiao7/DeepLOB/run_deeplob_pytorch_executed.ipynb

echo "Executing notebook: $NB_IN"
jupyter nbconvert \
    --to notebook \
    --execute \
    --ExecutePreprocessor.timeout=28800 \
    --ExecutePreprocessor.kernel_name=python3 \
    --output "$NB_OUT" \
    "$NB_IN"

echo ""
echo "=== Job complete: $(date) ==="
echo "Output notebook: $NB_OUT"
ls -lh /ocean/projects/mth250011p/xxiao7/DeepLOB/results/
