#!/bin/bash
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

python3 -c "import tqdm, seaborn, torchinfo, statsmodels; print('Packages OK')"
python3 -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"

NB_IN=/ocean/projects/mth250011p/xxiao7/DeepLOB/run_deeplob_pytorch.ipynb
NB_OUT=/ocean/projects/mth250011p/xxiao7/DeepLOB/run_deeplob_pytorch_executed.ipynb

echo "Executing notebook (training skipped — loading saved results)..."
jupyter nbconvert \
    --to notebook \
    --execute \
    --ExecutePreprocessor.timeout=3600 \
    --ExecutePreprocessor.kernel_name=python3 \
    --output "$NB_OUT" \
    "$NB_IN"

echo ""
echo "=== Done: $(date) ==="
echo "Output: $NB_OUT"
ls -lh /ocean/projects/mth250011p/xxiao7/DeepLOB/results/
