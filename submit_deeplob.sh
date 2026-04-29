#!/bin/bash
# =============================================================================
# submit_deeplob.sh — SLURM GPU job: FI-2010 DeepLOB training
# =============================================================================
# This wrapper runs the FI-2010 classification pipeline only:
#   1. DeepLOB training on the five paper horizons
#   2. Factor / baseline / trading analysis
#
# Submit with:
#   sbatch submit_deeplob.sh
#
# Force retraining:
#   sbatch submit_deeplob.sh --force
#
# Common overrides:
#   FI_PROFILE=paper|legacy|adaptive
#   FI_RUN_TAG=<name>               # write models/results into models/<name>, results/<name>
#   FI_MONITOR=val_acc|val_loss|val_macro_f1|val_mcc
#   FI_LABEL_SMOOTHING=<float>
#   FI_CLASS_WEIGHT_MODE=none|balanced_sqrt
#   FI_GRAD_CLIP=<float>
# =============================================================================
#SBATCH --job-name=deeplob_repro
#SBATCH --partition=GPU-small
#SBATCH --account=mth250011p
#SBATCH --gres=gpu:v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=60G
#SBATCH --time=08:00:00
#SBATCH --export=ALL
#SBATCH --output=/ocean/projects/mth250011p/xxiao7/DeepLOB/logs/slurm_%j.out
#SBATCH --error=/ocean/projects/mth250011p/xxiao7/DeepLOB/logs/slurm_%j.err

set -e
mkdir -p /ocean/projects/mth250011p/xxiao7/DeepLOB/logs

echo "=== DeepLOB FI Job ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Started:   $(date)"
echo ""

# Avoid inherited conda state from the submitting shell.
unset CONDA_DEFAULT_ENV CONDA_EXE CONDA_PREFIX CONDA_PREFIX_1 CONDA_PROMPT_MODIFIER CONDA_PYTHON_EXE CONDA_SHLVL _CE_CONDA _CE_M
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

echo "Python:   $(which python3)"
echo "PYTHONPATH: $PYTHONPATH"
python3 -c "import torch; print('PyTorch:', torch.__version__)"
python3 -c "import torch; cuda=torch.cuda.is_available(); print('CUDA:', cuda, torch.cuda.get_device_name(0) if cuda else '')"
python3 -c "import tqdm, seaborn, torchinfo, statsmodels; print('Extra packages: OK')"
echo ""

BASE=/ocean/projects/mth250011p/xxiao7/DeepLOB
FI_HORIZONS=${FI_HORIZONS:-"0 1 2 3 4"}
FI_FORCE=${FI_FORCE:-1}
FI_PROFILE=${FI_PROFILE:-paper}
FI_RUN_TAG=${FI_RUN_TAG:-}
FI_MONITOR=${FI_MONITOR:-}
FI_LABEL_SMOOTHING=${FI_LABEL_SMOOTHING:-}
FI_CLASS_WEIGHT_MODE=${FI_CLASS_WEIGHT_MODE:-}
FI_GRAD_CLIP=${FI_GRAD_CLIP:-}
FORCE_ARGS=()
if [[ "${FI_FORCE}" != "0" ]]; then
    FORCE_ARGS+=(--force)
fi

if [[ -n "${FI_RUN_TAG}" ]]; then
    export FI_MODEL_DIR=${FI_MODEL_DIR:-"${BASE}/models/${FI_RUN_TAG}"}
    export FI_RESULT_DIR=${FI_RESULT_DIR:-"${BASE}/results/${FI_RUN_TAG}"}
else
    export FI_MODEL_DIR=${FI_MODEL_DIR:-"${BASE}/models"}
    export FI_RESULT_DIR=${FI_RESULT_DIR:-"${BASE}/results"}
fi
mkdir -p "${FI_MODEL_DIR}" "${FI_RESULT_DIR}"

if [[ "${FI_PROFILE}" != "paper" && "${FI_PROFILE}" != "legacy" && "${FI_PROFILE}" != "adaptive" ]]; then
    echo "Unsupported FI_PROFILE=${FI_PROFILE}. Use paper, legacy, or adaptive."
    exit 1
fi

echo "FI horizons: ${FI_HORIZONS}"
echo "FI force rerun: ${FI_FORCE}"
echo "FI profile: ${FI_PROFILE}"
echo "FI run tag: ${FI_RUN_TAG:-<default>}"
echo "FI monitor override: ${FI_MONITOR:-<profile default>}"
echo "FI label smoothing override: ${FI_LABEL_SMOOTHING:-<profile default>}"
echo "FI class-weight override: ${FI_CLASS_WEIGHT_MODE:-<profile default>}"
echo "FI grad-clip override: ${FI_GRAD_CLIP:-<profile default>}"
echo "FI model dir: ${FI_MODEL_DIR}"
echo "FI result dir: ${FI_RESULT_DIR}"

echo "=== Phase 1: DeepLOB training ==="
TRAIN_ARGS=(
    --epochs 100
    --batch-size 32
    --lr 1e-3
    --lookback 100
    --patience 20
    --min-epochs 20
    --weight-decay 1e-4
    --dropout 0.2
    --horizon-profile "${FI_PROFILE}"
    --horizons ${FI_HORIZONS}
)

if [[ -n "${FI_MONITOR}" ]]; then
    TRAIN_ARGS+=(--monitor "${FI_MONITOR}")
fi
if [[ -n "${FI_LABEL_SMOOTHING}" ]]; then
    TRAIN_ARGS+=(--label-smoothing "${FI_LABEL_SMOOTHING}")
fi
if [[ -n "${FI_CLASS_WEIGHT_MODE}" ]]; then
    TRAIN_ARGS+=(--class-weight-mode "${FI_CLASS_WEIGHT_MODE}")
fi
if [[ -n "${FI_GRAD_CLIP}" ]]; then
    TRAIN_ARGS+=(--grad-clip "${FI_GRAD_CLIP}")
fi

python3 "${BASE}/scripts/train_deeplob.py" \
    "${TRAIN_ARGS[@]}" \
    "${FORCE_ARGS[@]}" \
    "$@"

echo ""
echo "=== Phase 2: FI factor + baseline analysis ==="
python3 "${BASE}/scripts/analyze_fi2010.py" \
    "${FORCE_ARGS[@]}"

echo ""
echo "=== Job complete: $(date) ==="
ls -lh "${FI_RESULT_DIR}"
