#!/bin/bash
# =============================================================================
# submit_optiver.sh — SLURM GPU job: Optiver DeepLOBLite training
# =============================================================================
# Three-phase workflow:
#   Phase 1  Data preparation (CPU)   : scripts/prepare_optiver.py
#            Extracts 2-level LOB data, applies causal time_id-based
#            normalization, constructs mid-price direction labels, saves
#            per-stock .npz files.
#
#   Phase 2  Model training (GPU)     : scripts/train_optiver.py
#            Runs k={1,5,10} horizons with class-balanced loss, early stopping,
#            universal zero-shot, fine-tuned transfer, and specific-stock OOS.
#   Phase 3  Result analysis (CPU)    : scripts/analyze_optiver.py
#            Builds paper-style Figure 6/7/8/9 analogues for the notebook report.
#
# Submit with:
#   sbatch submit_optiver.sh
#
# To force re-processing and re-training:
#   sbatch submit_optiver.sh --force
#
# Monitor job:
#   squeue -u $USER
#   tail -f logs/slurm_optiver_<jobid>.out
# =============================================================================
#SBATCH --job-name=optiver_deeplob
#SBATCH --partition=GPU-small
#SBATCH --account=mth250011p
#SBATCH --gres=gpu:v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=60G
#SBATCH --time=08:00:00
#SBATCH --output=/ocean/projects/mth250011p/xxiao7/DeepLOB/logs/slurm_optiver_%j.out
#SBATCH --error=/ocean/projects/mth250011p/xxiao7/DeepLOB/logs/slurm_optiver_%j.err

set -e
mkdir -p /ocean/projects/mth250011p/xxiao7/DeepLOB/logs
mkdir -p /ocean/projects/mth250011p/xxiao7/DeepLOB/data/optiver_processed
mkdir -p /ocean/projects/mth250011p/xxiao7/DeepLOB/results/optiver

echo "=== Optiver DeepLOBLite Job ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Started:   $(date)"
echo ""

# ---- Environment ----
module purge
module load AI/pytorch_23.02-1.13.1-py3
module load gcc/13.3.1-p20240614

export PYUSER=/ocean/projects/mth250011p/xxiao7/pyuser
export PYTHONUSERBASE=$PYUSER
export PIP_CACHE_DIR=/ocean/projects/mth250011p/xxiao7/pip_cache
export HF_HOME=/ocean/projects/mth250011p/xxiao7/huggingface-cache
export MPLCONFIGDIR=/ocean/projects/mth250011p/xxiao7/mpl_cache
mkdir -p $MPLCONFIGDIR

PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
export PYTHONPATH=$PYUSER/lib/python${PYVER}/site-packages:${PYTHONPATH:-}

echo "Python:     $(which python3)"
echo "PYTHONPATH: $PYTHONPATH"
python3 -c "import torch; print('PyTorch:', torch.__version__)"
python3 -c "import torch; cuda=torch.cuda.is_available(); print('CUDA:', cuda, torch.cuda.get_device_name(0) if cuda else '')"
python3 -c "import pyarrow, pandas; print('Parquet stack:', pyarrow.__version__, pandas.__version__)"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-<empty>}"
echo ""

BASE=/ocean/projects/mth250011p/xxiao7/DeepLOB

# ---- Phase 1: Data preparation (CPU-bound, runs on the GPU node) ----
echo "=== Phase 1: Data Preparation ==="
PREPARED_FLAG="${BASE}/data/optiver_processed/stock_split.json"
if [ ! -f "$PREPARED_FLAG" ] || echo "$@" | grep -q "\-\-force"; then
    echo "Running prepare_optiver.py ..."
    python3 "${BASE}/scripts/prepare_optiver.py" \
        --zip "${BASE}/optiver-realized-volatility-prediction.zip" \
        --out-dir "${BASE}/data/optiver_processed" \
        --alpha 0.002 \
        --norm-mode time-id \
        --norm-time-window 5 \
        --roll-norm 100 \
        --horizons 1 2 3 5 10 \
        $(echo "$@" | grep -q "\-\-force" && echo "--force")
else
    echo "Data already prepared (delete data/optiver_processed/ to redo)."
fi
echo ""

# ---- Phase 2: GPU training + transfer learning ----
echo "=== Phase 2: Model Training & Transfer Learning ==="
for HIDX in 0 3 4; do
    echo "--- Training horizon index ${HIDX} ---"
    python3 "${BASE}/scripts/train_optiver.py" \
        --epochs 30 \
        --transfer-epochs 12 \
        --batch-size 32 \
        --lr 2e-4 \
        --transfer-lr 5e-5 \
        --weight-decay 1e-4 \
        --patience 6 \
        --min-epochs 6 \
        --specific-epochs 25 \
        --lookback 50 \
        --horizon "${HIDX}" \
        --max-transfer-stocks 5 \
        "$@"
done

echo ""
echo "=== Phase 3: Paper-style analysis ==="
python3 "${BASE}/scripts/analyze_optiver.py" \
    --lookback 50 \
    --horizons 1 5 10 \
    --num-lime-samples 20 \
    --lime-time-bins 5 \
    --lime-feature-bins 2

echo ""
echo "=== Job complete: $(date) ==="
echo "Results:"
ls -lh "${BASE}/results/optiver/"
