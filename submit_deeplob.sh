#!/bin/bash
# =============================================================================
# submit_deeplob.sh — SLURM GPU job: FI-2010 DeepLOB training
# =============================================================================
# Heavy training runs here on GPU. The report notebook only loads saved outputs.
#
# Submit with:
#   sbatch submit_deeplob.sh
#
# Force retraining:
#   sbatch submit_deeplob.sh --force
# =============================================================================
#SBATCH --job-name=deeplob_repro
#SBATCH --partition=GPU-small
#SBATCH --account=mth250011p
#SBATCH --gres=gpu:v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=60G
#SBATCH --time=08:00:00
#SBATCH --output=/ocean/projects/mth250011p/xxiao7/DeepLOB/logs/slurm_%j.out
#SBATCH --error=/ocean/projects/mth250011p/xxiao7/DeepLOB/logs/slurm_%j.err

set -e
mkdir -p /ocean/projects/mth250011p/xxiao7/DeepLOB/logs

echo "=== DeepLOB Training Job ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Started:   $(date)"
echo ""

module purge
module load AI/pytorch_23.02-1.13.1-py3

export PYUSER=/ocean/projects/mth250011p/xxiao7/pyuser
export PYTHONUSERBASE=$PYUSER
export PIP_CACHE_DIR=/ocean/projects/mth250011p/xxiao7/pip_cache
export HF_HOME=/ocean/projects/mth250011p/xxiao7/huggingface-cache
export MPLCONFIGDIR=/ocean/projects/mth250011p/xxiao7/mpl_cache
mkdir -p $MPLCONFIGDIR

PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
export PYTHONPATH=$PYUSER/lib/python${PYVER}/site-packages:${PYTHONPATH:-}

echo "Python:   $(which python3)"
echo "PYTHONPATH: $PYTHONPATH"
python3 -c "import torch; print('PyTorch:', torch.__version__)"
python3 -c "import torch; cuda=torch.cuda.is_available(); print('CUDA:', cuda, torch.cuda.get_device_name(0) if cuda else '')"
python3 -c "import tqdm, seaborn, torchinfo, statsmodels; print('Extra packages: OK')"
echo ""

BASE=/ocean/projects/mth250011p/xxiao7/DeepLOB

python3 "${BASE}/scripts/train_deeplob.py" \
    --epochs 100 \
    --batch-size 32 \
    --lr 1e-3 \
    --lookback 100 \
    --patience 20 \
    --min-epochs 20 \
    --weight-decay 1e-4 \
    --dropout 0.2 \
    "$@"

echo ""
echo "=== Job complete: $(date) ==="
ls -lh "${BASE}/results/"
