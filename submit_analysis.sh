#!/bin/bash
# =============================================================================
# submit_analysis.sh — SLURM CPU job: generate FI-2010 analysis artifacts
# =============================================================================
# Assumes DeepLOB model weights / predictions already exist (submit_deeplob.sh).
# Runs scripts/analyze_fi2010.py to generate factor-analysis tables, baseline
# comparison outputs, and strategy statistics used by the report notebook.
# =============================================================================
#SBATCH --job-name=deeplob_analysis
#SBATCH --partition=RM-shared
#SBATCH --account=mth250011p
#SBATCH --cpus-per-task=16
#SBATCH --mem=31G
#SBATCH --time=01:00:00
#SBATCH --output=/ocean/projects/mth250011p/xxiao7/DeepLOB/logs/slurm_analysis_%j.out
#SBATCH --error=/ocean/projects/mth250011p/xxiao7/DeepLOB/logs/slurm_analysis_%j.err

set -e
mkdir -p /ocean/projects/mth250011p/xxiao7/DeepLOB/logs

echo "=== DeepLOB Analysis Job ==="
echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)  Started: $(date)"

module purge
module load AI/pytorch_23.02-1.13.1-py3

export PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
export PYTHONUSERBASE=/ocean/projects/mth250011p/xxiao7/pyuser
export PYTHONPATH=/ocean/projects/mth250011p/xxiao7/pyuser/lib/python${PYVER}/site-packages:${PYTHONPATH:-}
export PIP_CACHE_DIR=/ocean/projects/mth250011p/xxiao7/pip_cache
export MPLCONFIGDIR=/ocean/projects/mth250011p/xxiao7/mpl_cache
mkdir -p $MPLCONFIGDIR

python3 -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"

SCRIPT=/ocean/projects/mth250011p/xxiao7/DeepLOB/scripts/analyze_fi2010.py
echo "Running FI-2010 analysis script: $SCRIPT"
python3 "$SCRIPT"

echo ""
echo "=== Done: $(date) ==="
echo "Artifacts written under: /ocean/projects/mth250011p/xxiao7/DeepLOB/results"
