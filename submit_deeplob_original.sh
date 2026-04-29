#!/bin/bash
# =============================================================================
# submit_deeplob_original.sh — SLURM GPU job: exact original DeepLOB notebook
# =============================================================================
# Reproduces the original GitHub PyTorch notebook configuration on FI-2010.
#
# Submit with:
#   sbatch submit_deeplob_original.sh
# =============================================================================
#SBATCH --job-name=deeplob_orig
#SBATCH --partition=GPU-small
#SBATCH --account=mth250011p
#SBATCH --gres=gpu:v100-32:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --export=ALL
#SBATCH --output=/ocean/projects/mth250011p/xxiao7/DeepLOB/logs/slurm_orig_%j.out
#SBATCH --error=/ocean/projects/mth250011p/xxiao7/DeepLOB/logs/slurm_orig_%j.err

set -e
mkdir -p /ocean/projects/mth250011p/xxiao7/DeepLOB/logs

echo "=== DeepLOB Original Notebook Job ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Started:   $(date)"
echo ""

unset CONDA_DEFAULT_ENV CONDA_EXE CONDA_PREFIX CONDA_PREFIX_1 CONDA_PROMPT_MODIFIER CONDA_PYTHON_EXE CONDA_SHLVL _CE_CONDA _CE_M
module load AI/pytorch_23.02-1.13.1-py3
module load gcc/13.3.1-p20240614

export PYUSER=/ocean/projects/mth250011p/xxiao7/pyuser
export PYTHONUSERBASE=$PYUSER
export PIP_CACHE_DIR=/ocean/projects/mth250011p/xxiao7/pip_cache
export HF_HOME=/ocean/projects/mth250011p/xxiao7/huggingface-cache
export MPLCONFIGDIR=/ocean/projects/mth250011p/xxiao7/mpl_cache
mkdir -p "$MPLCONFIGDIR"

PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
export PYTHONPATH=$PYUSER/lib/python${PYVER}/site-packages:${PYTHONPATH:-}

echo "Python:   $(which python3)"
echo "PYTHONPATH: $PYTHONPATH"
python3 -c "import torch; print('PyTorch:', torch.__version__)"
python3 -c "import torch; cuda=torch.cuda.is_available(); print('CUDA:', cuda, torch.cuda.get_device_name(0) if cuda else '')"
echo ""

BASE=/ocean/projects/mth250011p/xxiao7/DeepLOB
export FI_ORIGINAL_MODEL_DIR=${FI_ORIGINAL_MODEL_DIR:-"${BASE}/models/fi_original_notebook"}
export FI_ORIGINAL_RESULT_DIR=${FI_ORIGINAL_RESULT_DIR:-"${BASE}/results/fi_original_notebook"}
mkdir -p "${FI_ORIGINAL_MODEL_DIR}" "${FI_ORIGINAL_RESULT_DIR}"

echo "Original notebook model dir: ${FI_ORIGINAL_MODEL_DIR}"
echo "Original notebook result dir: ${FI_ORIGINAL_RESULT_DIR}"
echo ""

python3 "${BASE}/scripts/train_deeplob_original_notebook.py"

echo ""
echo "=== Job complete: $(date) ==="
ls -lh "${FI_ORIGINAL_RESULT_DIR}"