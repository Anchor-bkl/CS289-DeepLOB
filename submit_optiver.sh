#!/bin/bash
# =============================================================================
# submit_optiver.sh — SLURM GPU job: classification-only Optiver training/save configuration
# =============================================================================
# This wrapper runs one classification-only Optiver job submission:
#   1. prepare_optiver.py   -> processed stock files / split metadata
#   2. train_optiver.py     -> retained k=5 transfer workflow
#   3. optionally analyze_optiver.py when OPTIVER_RUN_ANALYSIS=1
#
# Submit Original 3 with:
#   sbatch submit_optiver.sh
#
# Submit rolling 3 with:
#   sbatch --export=ALL,OPTIVER_LABEL_MODE=rolling-quantile-3class submit_optiver.sh
#
# Force a clean rerun of preprocessing and training:
#   sbatch submit_optiver.sh --force
# =============================================================================
#SBATCH --job-name=optiver_final
#SBATCH --partition=GPU-small
#SBATCH --account=mth250011p
#SBATCH --gres=gpu:v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=60G
#SBATCH --time=08:00:00
#SBATCH --export=ALL
#SBATCH --output=/ocean/projects/mth250011p/xxiao7/DeepLOB/logs/slurm_optiver_%j.out
#SBATCH --error=/ocean/projects/mth250011p/xxiao7/DeepLOB/logs/slurm_optiver_%j.err

set -e
mkdir -p /ocean/projects/mth250011p/xxiao7/DeepLOB/logs
mkdir -p /ocean/projects/mth250011p/xxiao7/DeepLOB/data/optiver_processed
mkdir -p /ocean/projects/mth250011p/xxiao7/DeepLOB/results/optiver

echo "=== Optiver Final Job ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Started:   $(date)"
echo ""

# ---- Environment ----
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

echo "Python:     $(which python3)"
echo "PYTHONPATH: $PYTHONPATH"
python3 -c "import torch; print('PyTorch:', torch.__version__)"
python3 -c "import torch; cuda=torch.cuda.is_available(); print('CUDA:', cuda, torch.cuda.get_device_name(0) if cuda else '')"
python3 -c "import pyarrow, pandas; print('Parquet stack:', pyarrow.__version__, pandas.__version__)"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-<empty>}"
echo ""

BASE=/ocean/projects/mth250011p/xxiao7/DeepLOB
FORCE_ARGS=()
if echo "$@" | grep -q "\-\-force"; then
    FORCE_ARGS+=(--force)
fi

OPTIVER_SAMPLE_MODE=volatility
OPTIVER_TRANSFER_STOCK_SELECTOR=${OPTIVER_TRANSFER_STOCK_SELECTOR:-balanced}
OPTIVER_MAX_TRANSFER_STOCKS=${OPTIVER_MAX_TRANSFER_STOCKS:-3}
OPTIVER_TRANSFER_STOCK_IDS=${OPTIVER_TRANSFER_STOCK_IDS:-}
OPTIVER_BASE_STOCK_SCOPE=${OPTIVER_BASE_STOCK_SCOPE:-selected-transfer-only}
OPTIVER_TRANSFER_TAIL_FRAC=${OPTIVER_TRANSFER_TAIL_FRAC:-0.30}
OPTIVER_TRANSFER_MODE=${OPTIVER_TRANSFER_MODE:-lstm_conv3}
OPTIVER_LABEL_MODE=${OPTIVER_LABEL_MODE:-original}
OPTIVER_QUANTILE_STATIONARY=${OPTIVER_QUANTILE_STATIONARY:-}
OPTIVER_ROLLING_QUANTILE_WINDOW=${OPTIVER_ROLLING_QUANTILE_WINDOW:-}
OPTIVER_TASK_TYPE=${OPTIVER_TASK_TYPE:-classification}
OPTIVER_RUN_ANALYSIS=${OPTIVER_RUN_ANALYSIS:-0}
OPTIVER_MODEL_DIR=${OPTIVER_MODEL_DIR:-}
OPTIVER_RESULT_DIR=${OPTIVER_RESULT_DIR:-}

if [[ "${OPTIVER_TASK_TYPE}" != "classification" ]]; then
    echo "submit_optiver.sh is classification-only. Unsupported OPTIVER_TASK_TYPE=${OPTIVER_TASK_TYPE}."
    exit 1
fi

OPTIVER_MONITOR=val_macro_f1

sanitize_optiver_tag() {
    printf '%s' "$1" | tr -c 'A-Za-z0-9._-' '_'
}

default_optiver_artifact_subdir() {
    case "$1" in
        original)
            printf 'classification_original3'
            ;;
        rolling-quantile-3class)
            printf 'classification_rolling3_w%s' "${OPTIVER_EFFECTIVE_ROLLING_WINDOW}"
            ;;
        stock-quantile)
            printf 'classification_stock_quantile'
            ;;
        original-5class)
            printf 'classification_original5'
            ;;
        rolling-quintile-5class)
            printf 'classification_rolling5_w%s' "${OPTIVER_EFFECTIVE_ROLLING_WINDOW}"
            ;;
        *)
            printf 'classification_%s' "$(sanitize_optiver_tag "$1")"
            ;;
    esac
}

OPTIVER_EFFECTIVE_ROLLING_WINDOW=${OPTIVER_ROLLING_QUANTILE_WINDOW:-20000}
OPTIVER_ARTIFACT_SUBDIR=$(default_optiver_artifact_subdir "${OPTIVER_LABEL_MODE}")
export OPTIVER_MODEL_DIR=${OPTIVER_MODEL_DIR:-"${BASE}/models/optiver/${OPTIVER_ARTIFACT_SUBDIR}"}
export OPTIVER_RESULT_DIR=${OPTIVER_RESULT_DIR:-"${BASE}/results/optiver/${OPTIVER_ARTIFACT_SUBDIR}"}
mkdir -p "${OPTIVER_MODEL_DIR}" "${OPTIVER_RESULT_DIR}"

echo "Retained Optiver horizon: k=5"
echo "Retained Optiver task: classification"
echo "Retained Optiver monitor: ${OPTIVER_MONITOR}"
echo "Retained Optiver sample mode: ${OPTIVER_SAMPLE_MODE}"
echo "Retained Optiver transfer selector: ${OPTIVER_TRANSFER_STOCK_SELECTOR}"
echo "Retained Optiver max transfer stocks: ${OPTIVER_MAX_TRANSFER_STOCKS}"
echo "Retained Optiver base stock scope: ${OPTIVER_BASE_STOCK_SCOPE}"
echo "Retained Optiver transfer tail frac: ${OPTIVER_TRANSFER_TAIL_FRAC}"
echo "Retained Optiver transfer mode: ${OPTIVER_TRANSFER_MODE}"
echo "Retained Optiver label mode: ${OPTIVER_LABEL_MODE}"
echo "Retained Optiver stationary-quantile override: ${OPTIVER_QUANTILE_STATIONARY:-<mode default>}"
echo "Retained Optiver rolling-quantile-window: ${OPTIVER_ROLLING_QUANTILE_WINDOW:-${OPTIVER_EFFECTIVE_ROLLING_WINDOW}}"
echo "Retained Optiver analysis phase: ${OPTIVER_RUN_ANALYSIS}"
echo "Retained transfer stocks: ${OPTIVER_TRANSFER_STOCK_IDS:-<auto>}"
echo "Optiver model dir: ${OPTIVER_MODEL_DIR}"
echo "Optiver result dir: ${OPTIVER_RESULT_DIR}"

# ---- Phase 1: Data preparation (CPU-bound, runs on the GPU node) ----
echo "=== Phase 1: Data Preparation ==="
PREPARED_FLAG="${BASE}/data/optiver_processed/stock_split.json"
if [ ! -f "$PREPARED_FLAG" ] || [[ ${#FORCE_ARGS[@]} -gt 0 ]]; then
    echo "Running prepare_optiver.py ..."
    python3 "${BASE}/scripts/prepare_optiver.py" \
        --zip "${BASE}/optiver-realized-volatility-prediction.zip" \
        --out-dir "${BASE}/data/optiver_processed" \
        --alpha 0.002 \
        --norm-mode event \
        --norm-time-window 5 \
        --norm-clip 12 \
        --roll-norm 100 \
        --num-transfer-stocks 10 \
        --split-mode interleaved \
        --split-seed 42 \
        --horizons 1 2 3 5 10 \
        "${FORCE_ARGS[@]}"
else
    echo "Data already prepared (delete data/optiver_processed/ to redo)."
fi
echo ""

# ---- Phase 2: GPU training + transfer learning ----
echo "=== Phase 2: Model Training & Transfer Learning ==="
TRAIN_ARGS=(
    --task-type classification
    --monitor "${OPTIVER_MONITOR}"
    --base-sample-mode "${OPTIVER_SAMPLE_MODE}"
    --base-stock-scope "${OPTIVER_BASE_STOCK_SCOPE}"
    --transfer-tail-frac "${OPTIVER_TRANSFER_TAIL_FRAC}"
    --transfer-mode "${OPTIVER_TRANSFER_MODE}"
    --transfer-stock-selector "${OPTIVER_TRANSFER_STOCK_SELECTOR}"
    --max-transfer-stocks "${OPTIVER_MAX_TRANSFER_STOCKS}"
)

if [[ -n "${OPTIVER_TRANSFER_STOCK_IDS}" ]]; then
    read -r -a OPTIVER_TRANSFER_STOCK_ID_ARRAY <<< "${OPTIVER_TRANSFER_STOCK_IDS}"
    TRAIN_ARGS+=(--transfer-stock-ids "${OPTIVER_TRANSFER_STOCK_ID_ARRAY[@]}")
fi
if [[ -n "${OPTIVER_LABEL_MODE}" ]]; then
    TRAIN_ARGS+=(--label-mode "${OPTIVER_LABEL_MODE}")
fi
if [[ -n "${OPTIVER_QUANTILE_STATIONARY}" ]]; then
    TRAIN_ARGS+=(--quantile-stationary "${OPTIVER_QUANTILE_STATIONARY}")
fi
if [[ -n "${OPTIVER_ROLLING_QUANTILE_WINDOW}" ]]; then
    TRAIN_ARGS+=(--rolling-quantile-window "${OPTIVER_ROLLING_QUANTILE_WINDOW}")
fi

python3 "${BASE}/scripts/train_optiver.py" \
    "${TRAIN_ARGS[@]}" \
    "${FORCE_ARGS[@]}"

echo ""
if [[ "${OPTIVER_RUN_ANALYSIS}" == "1" ]]; then
    echo "=== Phase 3: Paper-style analysis ==="
    python3 "${BASE}/scripts/analyze_optiver.py" \
        --lookback 50 \
        --horizons 5 \
        --num-lime-samples 20 \
        --lime-time-bins 5 \
        --lime-feature-bins 2
else
    echo "=== Phase 3: Analysis skipped (training/save only) ==="
fi

echo ""
echo "=== Job complete: $(date) ==="
echo "Results:"
ls -lh "${OPTIVER_RESULT_DIR}"
